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
from llama_index.core.workflow import Event
"""--------------------------------------------------------------------------------"""
# defining the events
class FirstEvent(Event):
    first_output: str

class SecondEvent(Event):
    second_output: str

# defining the workflows
class MyWorkflow(Workflow):
    # declare a function as a step
    @step
    async def step_one(self, ev: StartEvent) -> FirstEvent:
        # do something here
        #print (ev.first_input)
        #print (ev.first_output)
        return FirstEvent(first_output="First step complete!")
    
    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        # do something here
        # print (ev.first_input)
        return SecondEvent(second_output="Second step complete!")

    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        # do something here
        return StopEvent(third_output="Third step complete!")


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