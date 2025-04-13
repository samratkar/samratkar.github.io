from IPython.display import display, HTML
import random
import global_settings as gs
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
from llama_index.utils.workflow import draw_all_possible_flows
"""--------------------------------------------------------------------------------"""
class MyWorkflow(Workflow):
    # declare a function as a step
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")


async def main():
    gs.initialize_settings()
    api_key = gs.get_openai_api_key()
    basic_workflow = MyWorkflow(timeout=10, verbose=False)
    result = await basic_workflow.run()
    print(result)
    draw_all_possible_flows(basic_workflow, filename="C:/github/samratkar.github.io/_posts/concepts/explain/explain5/workflows/basic_workflow.html")

"""--------------------------------------------------------------------------------"""

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())