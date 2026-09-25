"""v3 entry; shares audited offline/live mechanics, with an isolated protocol."""
import evaluate_jev_rcta_adaptive_v2 as runner
import jev_rcta_adaptive_v3 as method

SOURCES = ("rcta_retrieval.py", "jev_rcta_adaptive_v3.py", "rcta_demo_v3.py", "evaluate_jev_rcta_adaptive_v3.py")


def main():
    runner.main(engine=method, additional_sources=SOURCES, demo_module="rcta_demo_v3")


if __name__ == "__main__":
    main()
