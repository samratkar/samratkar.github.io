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
class MyStartEvent(StartEvent):
    output: str
    output = "Reporting from Start Event"
    print ("This is inside the Start Event")
class FirstEvent(Event):
    output: str
    output = "Reporting from First Event"
    print ("This is inside the First Event")

class SecondEvent(Event):
    output: str
    output = "Reporting from Second Event"
    print ("This is inside the Second Event")

class MyStopEvent(StopEvent):
    output: str
    output = "Reporting from Stop Event"
    print ("This is inside the Stop Event")

class LoopEvent(Event):
    output: str
    output = "Reporting from Loop Event"
    print ("This is inside the Loop Event")

# defining the workflows
class MyWorkflow(Workflow):
    # declare a function as a step
    @step
    async def step_one(self, ev: MyStartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print ("Bad things happened in step 1. Looping now")
            return LoopEvent(output="Looping back to step 1")
        else:
            print ("Good things happened in step 1. Continuing now")
            print (f"This is step 1. Output from StartEvent = input to  FirstEvent: {ev.output}")
            return FirstEvent(output = "step 1 completed")
    
    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        # do something here
        print (f"This is step 2. Output from FirstEvent = input to  SecondEvent: {ev.output}")
        return SecondEvent(output="step 2 completed")

    @step
    async def step_three(self, ev: SecondEvent) -> MyStopEvent:
        # do something here
        print (f"This is step 3. Output from SecondEvent = input to  ThirdEvent: {ev.output}")
        return StopEvent(output="step 3 completed. wnorkflow completed")


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