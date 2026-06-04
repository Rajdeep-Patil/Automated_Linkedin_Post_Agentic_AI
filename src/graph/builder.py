from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.graph.state import AgentState
from src.graph.routers import chat_or_post_router,generate_post_tool_router,post_score_router
from src.nodes.chat_or_post import ChatOrPostNode
from src.nodes.chat import ChatNode
from src.nodes.generate_post import GeneratePostNode
from src.nodes.post_score import PostScoreNode
from src.nodes.regenerate_post import RegeneratePostNode
from src.exception.exception import AutomatedLinkedinPostAgent
from src.nodes.linkedin_post_tool import LinkedInToolNode
from langgraph.prebuilt import ToolNode, tools_condition
from src.logging.logger import logger
import sys
import os
import asyncio
import selectors
from dotenv import load_dotenv
load_dotenv()

class GraphBuilder:
    def __init__(self, model, model_with_both_tools,search_tools, linkedin_tools):  
        try:
            logger.info("Initializing GraphBuilder...")
            self.search_tools = search_tools
            self.linkedin_tools = linkedin_tools
            self.linkedin_tool_node = LinkedInToolNode(linkedin_tools=linkedin_tools)
            self.builder = StateGraph(AgentState)
            self.chat_or_post_node = ChatOrPostNode(model=model)
            self.chat_node = ChatNode(model=model,model_with_both_tools=model_with_both_tools)
            self.generate_post_node = GeneratePostNode(model=model,model_with_both_tools=model_with_both_tools)
            self.post_score_node = PostScoreNode(model=model)
            self.regenerate_post_node = RegeneratePostNode(model=model)
            logger.info("All nodes initialized successfully")

        except Exception as e:
            logger.exception("GraphBuilder initialization failed")
            raise AutomatedLinkedinPostAgent(e, sys)

    def build(self):
        try:
            logger.info("Building graph...")

            self.builder.add_node("chat_or_post",self.chat_or_post_node.chat_or_post)
            self.builder.add_node("chat",self.chat_node.chat_node)
            self.builder.add_node("tools", ToolNode(self.search_tools + self.linkedin_tools))
            self.builder.add_node("generate_post",self.generate_post_node.generate_post)
            self.builder.add_node("post_generate_search_tools", ToolNode(self.search_tools))
            # self.builder.add_node("post_generate_linkedin_tool", ToolNode(self.linkedin_tools))
            self.builder.add_node("post_generate_linkedin_tool", self.linkedin_tool_node.linkedin_tool_node)
            self.builder.add_node("post_score",self.post_score_node.post_score)
            self.builder.add_node("regenerate_post",self.regenerate_post_node.regenerate_post)

            self.builder.add_edge(START,"chat_or_post")

            self.builder.add_conditional_edges("chat_or_post",chat_or_post_router,{"generate_post": "generate_post","chat": "chat"})
            self.builder.add_conditional_edges("chat", tools_condition)
            self.builder.add_edge("tools","chat")
            self.builder.add_conditional_edges("generate_post",generate_post_tool_router,{"post_generate_search_tools": "post_generate_search_tools","post_generate_linkedin_tool": "post_generate_linkedin_tool","post_score": "post_score"})
            self.builder.add_edge("post_generate_search_tools","generate_post")
            self.builder.add_edge("post_generate_linkedin_tool",END)
            self.builder.add_conditional_edges("post_score",post_score_router,{"regenerate_post": "regenerate_post",END: END})
            self.builder.add_edge("regenerate_post","post_score")

            logger.info("Graph built successfully")
            return self.builder

        except Exception as e:
            logger.exception("Graph build failed")
            raise AutomatedLinkedinPostAgent(e, sys)

    async def compile_and_run(self, run_callback):
        try:
            logger.info("Compiling graph...")
            builder = self.build()
            DB_URI = os.getenv("DB_URI")

            async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
                await checkpointer.setup()
                graph = builder.compile(
                    checkpointer=checkpointer,
                    interrupt_before=["post_generate_linkedin_tool"]
                )
                logger.info("Graph compiled successfully")
                await run_callback(graph)  

        except Exception as e:
            logger.exception("Graph compile failed")
            raise AutomatedLinkedinPostAgent(e, sys)
    
    