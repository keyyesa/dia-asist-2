"""Run with:  python -m unittest discover -s tests -v"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src import documents, vectorstore  # noqa: E402
from src import llm_client, orchestrator  # noqa: E402
from src.agents.json_utils import extract_json  # noqa: E402
from src.agents import report_analysis  # noqa: E402
from tests.helpers import FakeEmbedder, FakeLLM  # noqa: E402


class KBTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.docs = os.path.join(self.tmp, "docs")
        self.idx = os.path.join(self.tmp, "idx")
        os.makedirs(self.docs)
        self.embedder = FakeEmbedder()
        patcher = mock.patch.object(vectorstore, "get_embedder", return_value=self.embedder)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, name, text):
        with open(os.path.join(self.docs, name), "w", encoding="utf-8") as f:
            f.write(text)

    def train(self, **kw):
        return vectorstore.train(self.docs, self.idx, **kw)

    def status(self):
        return vectorstore.get_training_status(self.docs, self.idx)


class TrainingTests(KBTestCase):
    def test_first_training_builds_index_and_manifest(self):
        self.write("a.txt", "Metformin is a first line drug for type 2 diabetes. " * 30)
        self.write("b.md", "HbA1c measures average glucose over three months. " * 30)
        self.assertTrue(self.status()["pending"])
        s = self.train()
        self.assertTrue(s["full"] and s["changed"])
        self.assertEqual(sorted(s["added"]), ["a.txt", "b.md"])
        st = self.status()
        self.assertFalse(st["pending"])
        self.assertEqual({d["status"] for d in st["documents"]}, {"trained"})
        kb = vectorstore.KnowledgeBase(self.idx)
        hits = kb.search("metformin drug", k=2)
        self.assertEqual(hits[0]["source"], "a.txt")

    def test_incremental_only_embeds_new_file(self):
        self.write("a.txt", "Metformin first line therapy. " * 40)
        self.train()
        self.write("c.txt", "Retinopathy screening yearly eye exam. " * 40)
        st = self.status()
        self.assertEqual([d["name"] for d in st["documents"] if d["status"] == "new"], ["c.txt"])
        with mock.patch.object(self.embedder, "encode", wraps=self.embedder.encode) as enc:
            s = self.train()
        self.assertFalse(s["full"])
        self.assertEqual(s["added"], ["c.txt"])
        self.assertEqual(s["unchanged"], 1)
        embedded = sum(len(c.args[0]) for c in enc.call_args_list)
        self.assertEqual(embedded, s["n_chunks"] - vectorstore.load_manifest(self.idx)["files"]["a.txt"]["chunks"])
        kb = vectorstore.KnowledgeBase(self.idx)
        self.assertEqual(set(kb.sources), {"a.txt", "c.txt"})
        self.assertEqual(kb.search("eye exam retinopathy", 1)[0]["source"], "c.txt")
        self.assertEqual(kb.search("metformin therapy", 1)[0]["source"], "a.txt")

    def test_no_changes_is_a_noop(self):
        self.write("a.txt", "Insulin lowers blood glucose. " * 20)
        self.train()
        token = vectorstore.index_token(self.idx)
        s = self.train()
        self.assertFalse(s["changed"])
        self.assertEqual(vectorstore.index_token(self.idx), token)

    def test_modified_and_removed_files(self):
        self.write("a.txt", "Alpha topic about kidneys. " * 30)
        self.write("b.txt", "Beta topic about feet and neuropathy. " * 30)
        self.train()
        self.write("a.txt", "Completely different text about statins and cholesterol. " * 30)
        os.remove(os.path.join(self.docs, "b.txt"))
        st = self.status()
        self.assertEqual(st["documents"][0]["status"], "modified")
        self.assertEqual(st["removed"], ["b.txt"])
        s = self.train()
        self.assertEqual(s["removed"], ["b.txt"])
        kb = vectorstore.KnowledgeBase(self.idx)
        self.assertEqual(kb.sources, ["a.txt"])
        self.assertIn("statins", kb.search("statins cholesterol", 1)[0]["content"])
        self.assertFalse(self.status()["pending"])

    def test_signature_change_forces_full_retrain(self):
        self.write("a.txt", "Diet and exercise help. " * 40)
        self.train()
        with mock.patch.object(config, "CHUNK_SIZE", 300):
            self.assertTrue(self.status()["needs_full_retrain"])
            self.assertTrue(self.train()["full"])
            self.assertFalse(self.status()["needs_full_retrain"])

    def test_legacy_index_without_manifest_is_usable_then_upgraded(self):
        self.write("a.txt", "Legacy document text about glucose. " * 30)
        self.train()
        os.remove(os.path.join(self.idx, vectorstore.MANIFEST_FILE))
        st = self.status()
        self.assertTrue(st["legacy"] and st["pending"])
        self.assertEqual(len(vectorstore.KnowledgeBase(self.idx).search("glucose", 2)), 2)
        self.assertTrue(self.train()["full"])
        self.assertFalse(self.status()["pending"])

    def test_unreadable_and_empty_files_are_skipped_not_fatal(self):
        self.write("good.txt", "Good content about hypoglycemia. " * 20)
        self.write("empty.txt", "   \n  ")
        with open(os.path.join(self.docs, "broken.pdf"), "wb") as f:
            f.write(b"this is not a pdf")
        s = self.train()
        self.assertEqual(s["added"], ["good.txt"])
        self.assertEqual({n for n, _ in s["skipped"]}, {"empty.txt", "broken.pdf"})

    def test_deleting_all_documents_clears_index(self):
        self.write("a.txt", "Some text. " * 40)
        self.train()
        os.remove(os.path.join(self.docs, "a.txt"))
        s = self.train()
        self.assertEqual(s["n_chunks"], 0)
        self.assertFalse(self.status()["index_exists"])

    def test_progress_callback_is_monotonic_and_ends_at_one(self):
        self.write("a.txt", "Text about complications. " * 200)
        seen = []
        self.train(progress_cb=lambda f, m: seen.append(f))
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(seen, sorted(seen))

    def test_concurrent_training_is_rejected(self):
        self.write("a.txt", "Text. " * 50)
        vectorstore._train_lock.acquire()
        try:
            with self.assertRaises(vectorstore.TrainingBusyError):
                self.train()
        finally:
            vectorstore._train_lock.release()

    def test_readme_in_docs_folder_is_ignored(self):
        self.write("README.md", "This describes the folder, not diabetes.")
        self.write("a.txt", "Real content. " * 30)
        self.assertEqual(vectorstore.list_document_files(self.docs), ["a.txt"])


class UploadTests(KBTestCase):
    def test_add_replace_unchanged_and_delete(self):
        self.assertEqual(vectorstore.add_document_bytes("x.txt", b"one", self.docs), ("x.txt", "added"))
        self.assertEqual(vectorstore.add_document_bytes("x.txt", b"one", self.docs), ("x.txt", "unchanged"))
        self.assertEqual(vectorstore.add_document_bytes("x.txt", b"two", self.docs), ("x.txt", "replaced"))
        self.assertTrue(vectorstore.delete_document("x.txt", self.docs))
        self.assertFalse(vectorstore.delete_document("x.txt", self.docs))

    def test_rejects_bad_names_and_traversal(self):
        with self.assertRaises(ValueError):
            vectorstore.add_document_bytes("virus.exe", b"x", self.docs)
        with self.assertRaises(ValueError):
            vectorstore.add_document_bytes("README.md", b"x", self.docs)
        name, _ = vectorstore.add_document_bytes("../../evil.txt", b"x", self.docs)
        self.assertEqual(name, "evil.txt")
        self.assertTrue(os.path.exists(os.path.join(self.docs, "evil.txt")))
        with self.assertRaises(ValueError):
            vectorstore.delete_document("../x.txt", self.docs)

    def test_docx_and_pdf_roundtrip(self):
        import docx
        import pymupdf
        d = docx.Document()
        d.add_paragraph("Docx paragraph about hypertension in diabetes.")
        path = os.path.join(self.docs, "r.docx")
        d.save(path)
        self.assertIn("hypertension", documents.extract_file(path))
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 72), "PDF text about foot care")
        data = pdf.tobytes()
        self.assertIn("foot care", documents.extract_text("f.pdf", data))
        self.assertNotIn("[Page", documents.extract_text("f.pdf", data, page_markers=False))
        with self.assertRaises(ValueError):
            documents.extract_text("f.pdf", b"garbage")


class PipelineTests(KBTestCase):
    def setUp(self):
        super().setUp()
        self.write("a.txt", "Metformin is first line therapy for type 2 diabetes. " * 20)
        self.train()
        self.kb = vectorstore.KnowledgeBase(self.idx)
        self.llm = FakeLLM()
        for target in ("generate", "generate_stream"):
            p = mock.patch.object(llm_client, target, getattr(self.llm, target))
            p.start()
            self.addCleanup(p.stop)
        orchestrator._report_cache.clear()

    def test_fast_mode_one_llm_call_streams_and_skips_verification(self):
        pipe = orchestrator.AgenticRAGPipeline(kb=self.kb)
        tokens, stages = [], []
        r = pipe.run("What is metformin?", mode="fast", on_token=tokens.append,
                     on_stage=lambda k, m: stages.append(k))
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual("".join(tokens).strip(), r["response"])
        self.assertEqual(r["verification_status"], "skipped")
        self.assertIsNone(r["verified"])
        self.assertEqual(stages, ["retrieve", "generate"])
        self.assertEqual(r["retrieved_sources"][0], "a.txt")

    def test_full_mode_runs_all_agents(self):
        pipe = orchestrator.AgenticRAGPipeline(kb=self.kb)
        r = pipe.run("What is metformin?", mode="full")
        self.assertTrue(r["verified"])
        self.assertTrue(r["optimized_query"].startswith("optimized:"))
        self.assertEqual(len(self.llm.calls), 3)  # optimize + generate + verify (no report => no router)

    def test_repeated_question_is_served_from_cache(self):
        pipe = orchestrator.AgenticRAGPipeline(kb=self.kb)
        pipe.run("What is metformin?", mode="fast")
        n = len(self.llm.calls)
        r = pipe.run("  what IS   metformin? ", mode="fast")
        self.assertTrue(r["cached"])
        self.assertEqual(len(self.llm.calls), n)
        self.assertFalse(pipe.run("What is metformin?", mode="full")["cached"])  # mode is part of the key
        pipe.clear_cache()
        self.assertFalse(pipe.run("What is metformin?", mode="fast")["cached"])

    def test_report_is_analysed_once_and_values_are_classified(self):
        pipe = orchestrator.AgenticRAGPipeline(kb=self.kb)
        report = "Marker A reading 7.8\nMarker B reading 142"  # unrecognisable names -> LLM fallback
        r = pipe.run("Explain my report", report_text=report, mode="fast")
        self.assertEqual(r["report_interpretation"]["HbA1c"]["classification"], "diabetes_range")
        self.assertEqual(r["report_interpretation"]["FBG"]["classification"], "diabetes_range")
        extraction_calls = [c for c in self.llm.calls if "Report Analysis" in c[1]]
        pipe.run("Another question about my report", report_text=report, mode="fast")
        self.assertEqual(len([c for c in self.llm.calls if "Report Analysis" in c[1]]), len(extraction_calls))

    def test_empty_query_and_bad_mode(self):
        pipe = orchestrator.AgenticRAGPipeline(kb=self.kb)
        with self.assertRaises(ValueError):
            pipe.run("   ")
        with self.assertRaises(ValueError):
            pipe.run("x", mode="turbo")

    def test_top_k_changes_number_of_sources(self):
        pipe = orchestrator.AgenticRAGPipeline(kb=self.kb)
        self.assertEqual(len(pipe.run("metformin", mode="fast", top_k=2)["retrieved_sources"]), 2)
        self.assertEqual(len(pipe.run("metformin", mode="fast", top_k=1)["retrieved_sources"]), 1)


SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_lab_reports")


def sample_text(name):
    with open(os.path.join(SAMPLES, name), "rb") as f:
        return documents.extract_text(name, f.read(), page_markers=False)


class LabReportRegressionTests(KBTestCase):
    """The reported bug: 'is my report normal?' / 'what is rbg from my report' got no answer for
    sample_report_2.txt (HbAlC typo, eAG, bare carriage returns, interpretation legend)."""

    def test_bare_carriage_returns_are_normalised(self):
        with open(os.path.join(SAMPLES, "sample_report_2.txt"), "rb") as f:
            raw = f.read()
        self.assertIn(b"\r", raw)
        self.assertNotIn("\r", documents.extract_text("x.txt", raw))

    def test_report_2_values_are_read_correctly(self):
        res = report_analysis.analyze_report(sample_text("sample_report_2.txt"))
        self.assertEqual(set(res), {"HbA1c", "eAG"})
        self.assertEqual(res["HbA1c"]["value"], 5.0)           # not 8 / 7 from the interpretation legend
        self.assertEqual(res["HbA1c"]["unit"], "%")
        self.assertEqual(res["HbA1c"]["classification"], "normal")
        self.assertEqual(res["HbA1c"]["lab_range"], "4.0 - 6.0")
        self.assertEqual(res["eAG"]["value"], 96.8)
        self.assertEqual(res["eAG"]["classification"], "normal")
        self.assertNotIn("RBG", res)

    def test_report_1_all_values(self):
        res = report_analysis.analyze_report(sample_text("sample_report_1.txt"))
        got = {k: (v["value"], v["classification"]) for k, v in res.items()}
        self.assertEqual(got, {
            "HbA1c": (7.8, "diabetes_range"), "FBG": (142.0, "diabetes_range"),
            "RBG": (210.0, "diabetes_range_if_symptomatic"), "Serum_Creatinine": (1.1, "normal"),
            "Urine_Albumin": (45.0, "microalbuminuria"), "LDL": (130.0, "borderline"),
            "HDL": (42.0, "acceptable"), "Triglycerides": (180.0, "borderline")})

    def test_extraction_variants_units_and_traps(self):
        f = report_analysis.analyze_report
        self.assertEqual(f("HbA1c 45 mmol/mol")["HbA1c"]["classification"], "prediabetes_range")  # ~6.3 %
        self.assertEqual(f("HbA1c 48 mmol/mol")["HbA1c"]["classification"], "diabetes_range")     # ~6.5 %
        self.assertEqual(f("Glycosylated Haemoglobin (HbA1c): 6.1 %")["HbA1c"]["classification"], "prediabetes_range")
        self.assertEqual(f("Fasting Blood Sugar (FBS) 5.5 mmol/L")["FBG"]["classification"], "normal")
        self.assertEqual(f("FBS: 5.0 mmol/L")["FBG"]["value"], 90.08)
        self.assertEqual(f("RBS 250 mg/dl")["RBG"]["classification"], "diabetes_range_if_symptomatic")
        self.assertEqual(f("Urine Creatinine 120 mg/dL\nSerum Creatinine 0.9 mg/dL")["Serum_Creatinine"]["value"], 0.9)
        self.assertEqual(f("HbA1c >8% action\nHbA1c 7-8% good"), {})   # thresholds are not results
        self.assertEqual(f("VLDL 20 mg/dL"), {})                        # VLDL is not LDL
        self.assertEqual(f("HDL 55 mg/dL\nLDL 90 mg/dL")["LDL"]["value"], 90.0)  # next test's value not stolen

    def test_missing_parameters_lists_what_is_absent(self):
        res = report_analysis.analyze_report(sample_text("sample_report_2.txt"))
        missing = report_analysis.missing_parameters(res)
        self.assertIn("random blood glucose (RBG)", missing)
        self.assertNotIn("HbA1c", missing)

    def test_router_sends_personal_report_questions_to_the_report_without_an_llm(self):
        from src.agents import router
        llm = FakeLLM()
        with mock.patch.object(llm_client, "generate", llm.generate):
            for q in ("is my report normal??", "what is rbg from my report", "Is the report normal?",
                      "are they normal?", "what does this result mean"):
                self.assertEqual(router.route(q, True), "BOTH", q)
        self.assertEqual(llm.calls, [])
        self.assertEqual(router.route("anything", False), "KNOWLEDGE_ONLY")

    def test_model_is_given_the_report_values_missing_tests_and_raw_text(self):
        self.write("a.txt", "HbA1c below 5.7 percent is normal. " * 20)
        self.train()
        pipe = orchestrator.AgenticRAGPipeline(kb=vectorstore.KnowledgeBase(self.idx))
        llm = FakeLLM()
        for target in ("generate", "generate_stream"):
            p = mock.patch.object(llm_client, target, getattr(llm, target))
            p.start()
            self.addCleanup(p.stop)
        report = sample_text("sample_report_2.txt")
        for mode in ("fast", "full"):
            llm.prompts.clear()
            pipe.run("what is rbg from my report", report_text=report, mode=mode, use_cache=False)
            answer_prompt = next(pr for sysm, pr in llm.prompts if "Response Generation Agent" in sysm)
            self.assertIn("HbA1c: 5.0 % -> normal (lab reference range: 4.0 - 6.0)", answer_prompt)
            self.assertIn("eAG: 96.8 mg/dL -> normal", answer_prompt)
            self.assertIn("Not found in this report:", answer_prompt)
            self.assertIn("random blood glucose (RBG)", answer_prompt.split("Not found in this report:")[1])
            self.assertIn("Estimated Average Glucose", answer_prompt)   # raw text is included too
        # rules found the values, so no LLM extraction call was needed (faster, deterministic)
        self.assertFalse(any("Report Analysis Agent" in sysm for sysm, _ in llm.prompts))

    def test_empty_analysis_is_not_cached(self):
        orchestrator._report_cache.clear()
        with mock.patch.object(report_analysis, "analyze_report", return_value={}):
            orchestrator.analyze_report_cached("nothing readable here")
        self.assertEqual(len(orchestrator._report_cache), 0)

    def test_patient_report_detection_for_uploads(self):
        with open(os.path.join(SAMPLES, "sample_report_2.txt"), "rb") as f:
            self.assertTrue(documents.looks_like_patient_report("lab report 2.txt", f.read()))
        self.assertFalse(documents.looks_like_patient_report("guide.txt", b"Metformin is first-line therapy."))


class MiscTests(unittest.TestCase):
    def test_extract_json_handles_fences_and_garbage(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json("no json here"), {})
        self.assertEqual(extract_json(None), {})

    def test_classification_and_unit_conversion(self):
        self.assertEqual(report_analysis.classify_value("HbA1c", 6.0), "prediabetes_range")
        self.assertAlmostEqual(report_analysis.convert_to_canonical("FBG", 7.0, "mmol/L"), 126.1, 1)

    def test_update_settings_validation(self):
        with self.assertRaises(ValueError):
            config.update_settings({"CHUNK_OVERLAP": 5000})
        with self.assertRaises(ValueError):
            config.update_settings({"LLM_BACKEND": "openai"})
        with self.assertRaises(KeyError):
            config.update_settings({"NOPE": 1})
        before = config.TOP_K
        config.update_settings({"TOP_K": 7})
        self.assertEqual(config.TOP_K, 7)
        config.update_settings({"TOP_K": before})

    def test_write_env_preserves_comments_and_updates_keys(self):
        with tempfile.TemporaryDirectory() as d:
            env = os.path.join(d, ".env")
            with open(env, "w") as f:
                f.write("# comment\nTOP_K=4\nOTHER=x\n")
            with mock.patch.object(config, "ENV_PATH", env):
                config._write_env({"TOP_K": 9, "OLLAMA_MODEL": "phi3:mini"})
            with open(env) as fh:
                text = fh.read()
            self.assertIn("# comment", text)
            self.assertIn("TOP_K=9", text)
            self.assertIn("OTHER=x", text)
            self.assertIn("OLLAMA_MODEL=phi3:mini", text)

    def test_ollama_streaming_parses_ndjson_and_connection_errors(self):
        class Resp:
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def iter_lines(self):
                yield b'{"response": "Hel"}'
                yield b""
                yield b'{"response": "lo", "done": false}'
                yield b'{"response": "", "done": true}'
        with mock.patch.object(llm_client._session, "post", return_value=Resp()):
            self.assertEqual("".join(llm_client._stream_ollama("p", "s", 0.1)), "Hello")
        import requests
        with mock.patch.object(llm_client._session, "post", side_effect=requests.ConnectionError()):
            with self.assertRaises(llm_client.LLMError) as cm:
                list(llm_client._stream_ollama("p", "s", 0.1))
            self.assertIn("ollama serve", str(cm.exception))
        with mock.patch.object(llm_client._session, "get", side_effect=requests.ConnectionError()):
            self.assertFalse(llm_client.check_connection()["ok"])


if __name__ == "__main__":
    unittest.main()
