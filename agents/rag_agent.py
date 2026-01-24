from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    
class RAGAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.tool_executor = ToolExecutor(tools)
        self.graph = self._create_graph()
    
    def _create_graph(self):
        """Create the agent graph"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # Add edge from tools to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _agent_node(self, state: AgentState):
        """Agent decision node"""
        messages = state["messages"]
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with access to a document search tool. "
                      "When users ask questions, first check if you should search the documents. "
                      "Use the search_documents tool to find relevant information from uploaded documents."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Get response
        response = llm_with_tools.invoke(messages)
        
        return {"messages": messages + [response]}
    
    def _tool_node(self, state: AgentState):
        """Execute tools"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Execute tools
        tool_calls = last_message.tool_calls
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Find and execute tool
            for tool in self.tools:
                if tool.name == tool_name:
                    result = tool.func(**tool_args)
                    results.append(AIMessage(content=result))
        
        return {"messages": messages + results}
    
    def _should_continue(self, state: AgentState):
        """Decide whether to continue or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        return "end"
    
    def run(self, user_input: str, chat_history: list):
        """Run the agent"""
        messages = []
        
        # Add chat history
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # Add new message
        messages.append(HumanMessage(content=user_input))
        
        # Run graph
        result = self.graph.invoke({"messages": messages})
        
        # Get final response
        final_message = result["messages"][-1]
        return final_message.content