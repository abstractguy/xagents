from xagents.cli import Executor


class TestExecutor:
    executor = Executor()

    def test_help(self, capsys, display_only_args):
        self.executor.maybe_create_agent(display_only_args[0])
        cap = capsys.readouterr().out
        for flag in display_only_args[1]:
            assert flag in cap
