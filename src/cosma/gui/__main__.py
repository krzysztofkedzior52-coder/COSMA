from .main_window import run

if __name__ == "__main__":          # the guard matters: worker processes (spawn) import this module again
    raise SystemExit(run())
