import chainlit as cl
from site_bot_opik.graph.graph import graph_builder
from opik.integrations.langchain import OpikTracer

@cl.on_message
async def on_message(message: cl.Message):
    content = message.content
    # opik config done. use in callback for graph invocation. you're set.
    # you'll need to setup your opik envs in .env file. see docs.
    opik_tracer = OpikTracer(graph=graph_builder.get_graph(xray=True))
    opik_config = {
        "callbacks": [opik_tracer]
    }
    
    final_response = f"You asked: {message.content}\nIntegrate w your RAG response here."
    # TODO: FIXME: sample code to build out your graph invocation stream
    # async for chunk in graph_builder.astream(
    #   {"messages": [{"role": "user", "content": content}]}, config=opik_config):
    #   print(f"chunk, \n{chunk}")
    
    #   # Each chunk is {node_name: node_output}
    #   for node_name, node_output in chunk.items():
    #       print(f"33333 - Node: {node_name}")
          
    #       # Check if this node produced messages
    #       if isinstance(node_output, dict) and "messages" in node_output:
    #           final_response = node_output["messages"][-1].content
    #           print(f"Got response: {final_response[:100]}...")
      
    await cl.Message(content=final_response).send()

# to prod users on what queries to write
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Starter Question 1",
            message="Starter Question 1",
            # icon="/public/idea.svg",
        ),

        cl.Starter(
            label="Starter Question 2",
            message="Starter Question 2",
            # icon="/public/learn.svg",
        ),
        cl.Starter(
            label="Starter Question 3",
            message="Starter Question 3",
            # icon="/public/write.svg",
        )
    ]
