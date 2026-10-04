"""Compare conversation continuity, cross-thread profile recall, and user isolation."""

from langchain_core.language_models.chat_models import BaseChatModel

from playground_agentic.example_support import model_from_env, show
from playground_agentic.scenarios.s03_memory_chat.graph import ChatTurn, build_graph, run_turn


def run_example(model: BaseChatModel) -> dict[str, object]:
    graph = build_graph(model)
    turns = [
        ChatTurn(
            user_id="demo-alex",
            thread_id="alex-1",
            message=(
                "My name is Alex. I prefer Python examples. Keep answers concise. "
                "We are designing a support ticket workflow."
            ),
        ),
        ChatTurn(
            user_id="demo-alex",
            thread_id="alex-1",
            message=("What workflow were we discussing? Suggest one next step."),
        ),
        ChatTurn(
            user_id="demo-alex",
            thread_id="alex-2",
            message=(
                "What preferences do you know about me? Do you know the previous topic? "
                "Do not guess facts missing from this thread or the supplied profile."
            ),
        ),
        ChatTurn(
            user_id="demo-sam",
            thread_id="sam-1",
            message=("Do you know my name or preferred programming language? Do not guess."),
        ),
    ]
    return {"turns": [run_turn(turn, compiled_graph=graph).model_dump() for turn in turns]}


if __name__ == "__main__":
    show(run_example(model_from_env()))
