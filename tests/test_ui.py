"""UI smoke tests: drives every page of the real app with Streamlit's AppTest.
Run with:  python -m unittest tests.test_ui -v"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src import llm_client, vectorstore  # noqa: E402
from tests.helpers import FakeEmbedder, FakeLLM  # noqa: E402

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Wrapper script: runs the real app.py, but makes a chosen page the default one
# (AppTest cannot click the sidebar menu, and only supports file-based page switching).
WRAPPER = f'''
import streamlit as st
_Page = st.Page
def _page(*args, **kwargs):
    target = st.session_state.get("_test_page")
    if target:
        kwargs["default"] = kwargs.get("url_path") == target
    return _Page(*args, **kwargs)
st.Page = _page
exec(compile(open(r"{os.path.join(ROOT, 'app.py')}", encoding="utf-8").read(), "app.py", "exec"))
'''

URL = {"dashboard": "dashboard", "chat": "chat", "lab": "lab-reports", "kb": "knowledge-base",
       "compare": "compare", "evaluation": "evaluation", "settings": "settings", "about": "about"}


@unittest.skipIf(AppTest is None, "streamlit AppTest not available")
class UITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.docs = os.path.join(self.tmp, "docs")
        self.idx = os.path.join(self.tmp, "idx")
        shutil.copytree(os.path.join(ROOT, "data", "knowledge_base_docs"), self.docs)
        self.llm = FakeLLM()
        patches = [
            mock.patch.object(config, "RAW_DOCS_DIR", self.docs),
            mock.patch.object(config, "FAISS_INDEX_DIR", self.idx),
            mock.patch.object(vectorstore, "get_embedder", return_value=FakeEmbedder()),
            mock.patch.object(llm_client, "generate", self.llm.generate),
            mock.patch.object(llm_client, "generate_stream", self.llm.generate_stream),
            mock.patch.object(llm_client, "check_connection",
                              return_value={"ok": True, "backend": "ollama", "model": "fake", "models": [], "detail": "ok"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def app(self, page=None):
        at = AppTest.from_string(WRAPPER, default_timeout=60)
        if page:
            at.session_state["_test_page"] = URL[page]
        return at.run()

    def assertClean(self, at):
        self.assertEqual([str(e.value) for e in at.exception], [])

    def train_via_ui(self):
        at = self.app("kb")
        self.assertClean(at)
        train = next(b for b in at.button if "Train" in b.label and "Full" not in b.label)
        at = train.click().run()
        self.assertClean(at)
        return at

    def test_every_page_renders_without_errors(self):
        for page in ("dashboard", "chat", "lab", "kb", "compare", "evaluation", "settings", "about"):
            with self.subTest(page=page):
                self.assertClean(self.app(page))

    def test_chat_is_blocked_until_trained_then_train_button_unlocks_it(self):
        at = self.app("chat")
        self.assertEqual(len(at.chat_input), 0)  # no chat box before training
        self.assertTrue(any("not been trained" in w.value for w in at.warning))
        at = self.train_via_ui()
        self.assertTrue(vectorstore.get_training_status()["index_exists"])
        self.assertFalse(vectorstore.get_training_status()["pending"])
        self.assertTrue(any("Training complete" in s.value for s in at.success))
        at = self.app("chat")
        self.assertClean(at)
        self.assertEqual(len(at.chat_input), 1)

    def test_chat_fast_stream_cache_and_full_mode(self):
        self.train_via_ui()
        at = self.app("chat")
        at.chat_input[0].set_value("What is HbA1c?").run()
        self.assertClean(at)
        self.assertEqual([m.name for m in at.chat_message], ["user", "assistant"])
        self.assertIn("Streamed answer", at.chat_message[1].markdown[0].value)
        n_calls = len(self.llm.calls)
        at.chat_input[0].set_value("what is hba1c?").run()
        self.assertClean(at)
        self.assertEqual(len(self.llm.calls), n_calls)  # second answer came from the cache
        self.assertEqual(len(at.chat_message), 4)
        at.sidebar.radio(key="mode").set_value("full").run()
        at.chat_input[0].set_value("How is diabetes diagnosed?").run()
        self.assertClean(at)
        self.assertEqual(len(at.chat_message), 6)
        captions = [c.value for c in at.caption]
        self.assertTrue(any("🧠 Full" in c and ("verifier" in c or "verified" in c) for c in captions), captions)
        self.assertTrue(any("⚡ Fast" in c for c in captions), captions)

    def test_new_document_shows_pending_banner_and_old_answers_still_work(self):
        self.train_via_ui()
        with open(os.path.join(self.docs, "new_doc.txt"), "w") as f:
            f.write("Statins reduce cholesterol in people with diabetes. " * 20)
        at = self.app("chat")
        self.assertClean(at)
        self.assertTrue(any("not trained yet" in w.value for w in at.warning))
        self.assertEqual(len(at.chat_input), 1)
        at = self.app("kb")
        at = next(b for b in at.button if b.label.startswith("🚀 Train")).click().run()
        self.assertClean(at)
        self.assertEqual(
            {d["status"] for d in vectorstore.get_training_status()["documents"]}, {"trained"})
        kb = vectorstore.KnowledgeBase()
        self.assertIn("new_doc.txt", kb.sources)

    def test_sidebar_train_button_and_full_retrain(self):
        at = self.app("dashboard")
        side = next(b for b in at.sidebar.button if "Train now" in b.label)
        at = side.click().run()
        self.assertClean(at)
        self.assertTrue(vectorstore.get_training_status()["index_exists"])
        at = self.app("kb")
        at = next(b for b in at.button if "Full retrain" in b.label).click().run()
        self.assertClean(at)
        self.assertTrue(any("Training complete" in s.value for s in at.success))

    def test_lab_report_flow(self):
        at = self.app("lab")
        next(b for b in at.button if "sample" in b.label).click().run()
        at = next(b for b in at.button if "Analyze" in b.label).click().run()
        self.assertClean(at)
        self.assertEqual(len(at.dataframe), 1)
        values = at.dataframe[0].value
        self.assertIn("HbA1c", list(values["Parameter"]))

    def test_compare_and_evaluation_and_settings(self):
        self.train_via_ui()
        at = self.app("compare")
        at.text_input[0].set_value("What is metformin?")
        at = next(b for b in at.button if "Compare" in b.label).click().run()
        self.assertClean(at)
        self.assertEqual(len(at.dataframe), 1)

        at = self.app("evaluation")
        at = next(b for b in at.button if "Run evaluation" in b.label).click().run()
        self.assertClean(at)
        self.assertTrue(len(at.dataframe) >= 1)

        at = self.app("settings")
        self.assertClean(at)
        at = next(b for b in at.button if b.key == "kb_apply").click().run()
        self.assertClean(at)

    def test_upload_auto_trains_and_chat_uses_the_new_document(self):
        self.train_via_ui()
        at = self.app("kb")
        content = b"Zorbaxin is a fictional drug used to treat nocturnal glucose spikes. " * 12
        at.file_uploader[0].set_value([("zorbaxin_notes.txt", content, "text/plain")])
        at = at.run()
        self.assertClean(at)
        status = vectorstore.get_training_status()
        self.assertFalse(status["pending"])  # auto-trained, nothing left waiting
        self.assertIn("zorbaxin_notes.txt", [d["name"] for d in status["documents"] if d["status"] == "trained"])
        self.assertTrue(any("Training complete" in s.value for s in at.success), [s.value for s in at.success])
        # the chatbot's answer is now grounded in the newly trained file
        chat = self.app("chat")
        chat.chat_input[0].set_value("What is zorbaxin?").run()
        self.assertClean(chat)
        details = " ".join(m.value for m in chat.expander[-1].markdown)
        self.assertIn("zorbaxin_notes.txt", details)

    def test_upload_without_autotrain_waits_for_train_button(self):
        self.train_via_ui()
        at = self.app("kb")
        at.checkbox(key="auto_train").set_value(False)
        at.file_uploader[0].set_value([("later.txt", b"Retinopathy eye screening guidance. " * 30, "text/plain")])
        at = at.run()
        self.assertClean(at)
        self.assertTrue(vectorstore.get_training_status()["pending"])  # saved but NOT trained
        self.assertNotIn("later.txt", vectorstore.KnowledgeBase().sources)
        train = next(b for b in at.button if b.label.startswith("🚀 Train"))
        at = train.click().run()
        self.assertClean(at)
        self.assertFalse(vectorstore.get_training_status()["pending"])
        self.assertIn("later.txt", vectorstore.KnowledgeBase().sources)

    def test_bad_upload_is_rejected_gracefully(self):
        at = self.app("kb")
        at.file_uploader[0].set_value([("notes.txt", b"", "text/plain")])  # empty file: saved, skipped on training
        at = at.run()
        self.assertClean(at)

    def _answer_prompts(self):
        return [pr for sysm, pr in self.llm.prompts if "Response Generation Agent" in sysm]

    def test_user_scenario_attach_report_2_then_ask_if_normal_and_about_rbg(self):
        self.train_via_ui()
        at = self.app("chat")
        with open(os.path.join(ROOT, "sample_lab_reports", "sample_report_2.txt"), "rb") as f:
            raw = f.read()
        uploader = next(u for u in at.file_uploader if "PDF, TXT or DOCX" in u.label)
        uploader.set_value([("lab report 2.txt", raw, "text/plain")])
        at = at.run()
        self.assertClean(at)
        self.assertTrue(any("Attached: lab report 2.txt" in s.value for s in at.success))
        for question in ("is my report normal??", "what is rbg from my report"):
            self.llm.prompts.clear()
            at.chat_input[0].set_value(question).run()
            self.assertClean(at)
            prompt = self._answer_prompts()[-1]
            self.assertIn("HbA1c: 5.0 % -> normal (lab reference range: 4.0 - 6.0)", prompt, question)
            self.assertIn("random blood glucose (RBG)", prompt.split("Not found in this report:")[1], question)
        # answers rendered, with the lab values table in the details
        self.assertEqual(len(at.chat_message), 4)
        tables = [df.value for df in at.dataframe]
        self.assertTrue(any("HbA1c" in list(t["Parameter"]) for t in tables))

    def test_asking_about_my_report_without_one_attached_explains_what_to_do(self):
        self.train_via_ui()
        at = self.app("chat")
        at.chat_input[0].set_value("is my report normal?").run()
        self.assertClean(at)
        self.assertTrue(any("No lab report is attached" in i.value for i in at.info))
        at.chat_input[0].set_value("What is HbA1c?").run()
        self.assertClean(at)
        # still shown once (kept in history for the personal question), and NOT added for the general one
        self.assertEqual(sum("No lab report is attached" in i.value for i in at.info), 1)

    def test_uploading_a_personal_lab_report_to_the_knowledge_base_warns(self):
        at = self.app("kb")
        with open(os.path.join(ROOT, "sample_lab_reports", "sample_report_2.txt"), "rb") as f:
            at.file_uploader[0].set_value([("lab report 2.txt", f.read(), "text/plain")])
        at = at.run()
        self.assertClean(at)
        self.assertTrue(any("looks like a personal lab report" in w.value for w in at.warning),
                        [w.value for w in at.warning])

    def test_lab_page_loads_each_sample_and_shows_lab_range(self):
        at = self.app("lab")
        at.selectbox(key="sample_choice").set_value("sample_report_2.txt")
        at = next(b for b in at.button if "Load sample" in b.label).click().run()
        at = next(b for b in at.button if "Analyze" in b.label).click().run()
        self.assertClean(at)
        df = at.dataframe[0].value
        self.assertEqual(set(df["Parameter"]), {"HbA1c", "eAG"})
        self.assertIn("4.0 - 6.0", list(df["Lab range"]))

    def test_llm_failure_is_shown_in_chat_not_as_crash(self):
        self.train_via_ui()
        at = self.app("chat")
        def boom(*a, **k):
            raise llm_client.LLMError("Cannot connect to Ollama")
            yield
        with mock.patch.object(llm_client, "generate_stream", boom):
            at.chat_input[0].set_value("anything new?").run()
        self.assertClean(at)
        self.assertTrue(any("Cannot connect to Ollama" in e.value for e in at.error))


if __name__ == "__main__":
    unittest.main()
