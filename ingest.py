"""CLI: train (index) the knowledge base without opening the app.

Usage:
    python ingest.py           # train only new/changed/removed documents
    python ingest.py --full    # rebuild the whole index from scratch
"""
import argparse
import sys

from src import vectorstore


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the DiaAssist knowledge base.")
    parser.add_argument("--full", action="store_true", help="rebuild the entire index")
    args = parser.parse_args()

    last = {"msg": ""}

    def show(frac, msg):
        if msg != last["msg"]:
            print(f"[{frac:4.0%}] {msg}")
            last["msg"] = msg

    summary = vectorstore.train(force_full=args.full, progress_cb=show)
    for name, reason in summary["skipped"]:
        print(f"Skipped {name}: {reason}")
    if summary["n_chunks"] == 0:
        print("\nNo readable documents found. Add .pdf/.txt/.md/.docx files to the documents folder "
              "(see data/knowledge_base_docs/README.md) and run again.")
        return 1
    if not summary["changed"]:
        print("\nNothing to do: everything is already trained.")
    else:
        print(f"\nDone. {len(summary['added'])} file(s) trained, {len(summary['removed'])} removed, "
              f"{summary['unchanged']} reused -> {summary['n_chunks']} chunks ({summary['seconds']}s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
