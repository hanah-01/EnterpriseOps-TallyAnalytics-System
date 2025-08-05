from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from .sub_agents.financial_analyst.agent import financial_analyst
from .sub_agents.inventory_analyst.agent import inventory_analyst
from .sub_agents.cost_management_analyst.agent import cost_management_analyst
from .sub_agents.tax_and_compliance_analyst.agent import tax_and_compliance_analyst
from .sub_agents.banking_analyst.agent import banking_analyst
from .tools.tools import get_current_time_tool

root_agent = Agent(
    name="manager",
    model="gemini-2.0-flash",
    description="Manager agent",
    instruction="""
    You are a manager agent that is responsible for overseeing the work of the other agents.

    Database Schema Overview:
    The tally.db database contains master tables (mst_*) and transaction tables (trn_*):
    Master Tables:
    1. mst_ledger (Chart of Accounts)
    2. mst_stock_item (Stock Items)
    3. mst_cost_centre (Cost Centers)
    Transaction Tables:
    1. trn_accounting (Accounting Entries)
    2. trn_inventory (Inventory Transactions)
    3. trn_bank (Bank Transactions)

    Always delegate the task to the appropriate agent. Use your best judgement to determine which agent to delegate to.

    You are responsible for delegating tasks to the following agents:
    - financial_analyst (mst_ledger, trn_accounting, trn_voucher)
    - inventory_analyst (mst_stock_item, trn_inventory, trn_batch)
    - cost_management_analyst (mst_cost_centre, trn_cost_centre)
    - tax_and_compliance_analyst (mst_gst_effective_rate)
    - banking_analyst (trn_bank)

    You also have access to the following tools:
    - get_current_time
    """,
    sub_agents=[
        financial_analyst,
        inventory_analyst,
        cost_management_analyst,
        tax_and_compliance_analyst,
        banking_analyst,
    ],
    tools=[
        get_current_time_tool,
    ],
)