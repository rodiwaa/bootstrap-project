import chainlit as cl
from site_bot_opik.graph.graph import graph_builder
from opik.integrations.langchain import OpikTracer

@cl.on_message
async def on_message(message: cl.Message):
    content = message.content
    
    # # Stream the graph output
    final_response = ""
    # print('1111111')
    # async for chunk in graph_builder.astream(
    #     {"messages": [{"role": "user", "content": content}]}
    # ):
    #   print('22222')
    #   print(f"chunk, \n{chunk}")
    #   if "messages" in chunk:
    #     print('33333')
    #     final_response = chunk["messages"][-1].content
    opik_tracer = OpikTracer(graph=graph_builder.get_graph(xray=True))
    opik_config = {
        "callbacks": [opik_tracer]
    }
    
    async for chunk in graph_builder.astream(
      {"messages": [{"role": "user", "content": content}]}, config=opik_config):
      print(f"chunk, \n{chunk}")
    
      # Each chunk is {node_name: node_output}
      for node_name, node_output in chunk.items():
          print(f"33333 - Node: {node_name}")
          
          # Check if this node produced messages
          if isinstance(node_output, dict) and "messages" in node_output:
              final_response = node_output["messages"][-1].content
              print(f"Got response: {final_response[:100]}...")
      
    print('444444')
    # Send the final answer to Chainlit UI
    # await cl.Message(content="whaddapp ya'lll").send()
    await cl.Message(content=final_response).send()
