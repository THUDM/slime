"""Prompt utilities for tau2-bench (dual-control)."""

from __future__ import annotations

import json
import os
from typing import Any


USE_COMPRESSED = os.environ.get("TAU2_USE_COMPRESSED_PROMPTS", "0") != "0"


COMPRESSED_POLICY_TELECOM = """# Telecom Agent Policy

## Capabilities
Help users with: technical support, overdue bill payment, line suspension, plan changes, data roaming, data refueling.

## Core Rules
1. Authenticate customer first (by phone, customer_id, or name+DOB).
2. One tool call OR one response per turn, never both.
3. Confirm before any state-changing action.
4. Transfer to human only if request is outside your capabilities.

## Action Ownership (CRITICAL)
You can directly perform: customer lookup, account queries, enable/disable roaming, resume lines, refuel data, process payments.

For device diagnostics and settings, you must INSTRUCT THE USER to perform them:
- Ask user to toggle airplane mode on/off
- Ask user to check and reset APN settings
- Ask user to restart/reboot their device
- Ask user to reseat their SIM card
- Ask user to toggle mobile data on/off
- Ask user to run a speed test

After instructing the user, wait for them to confirm they've done it, then proceed.

## Key Procedures

### Customer Lookup
- Look up by phone number, customer ID, or full name (requires DOB for verification).
- Get details for customer, line, device, bill, or plan as needed.

### Line Suspension
- Lines suspend for: overdue bills OR expired contract.
- Can resume after bills paid, UNLESS contract expired.
- After resuming a line, ask user to reboot their device.

### Overdue Bills
- Check bill status to confirm it is overdue.
- Only one bill can be AWAITING_PAYMENT at a time.
- Flow: send payment request -> user checks and accepts -> make payment -> verify PAID.

### Data Issues
- Check data usage vs plan limit.
- If exceeded: offer plan change OR data refuel (max 2GB).
- If abroad: check and enable roaming if needed.

### Technical Support
- First identify the customer and check their line/device status.
- For connectivity issues, instruct user step-by-step:
  1. Toggle airplane mode on, wait 10 seconds, toggle off
  2. If still not working, restart the device
  3. If still not working, check/reset APN settings
  4. If still not working, reseat SIM card
- After each step, ask user to confirm and check if issue is resolved.
- Only transfer to human after troubleshooting steps are exhausted.
"""

COMPRESSED_POLICY_RETAIL = """# Retail Agent Policy

## Capabilities
Help users: cancel/modify pending orders, return/exchange delivered orders, lookup order/product info.

## Core Rules
1. Authenticate user first (by email OR name+zip). Required even if user provides user_id.
2. One tool call OR one response per turn, never both.
3. Confirm before any database update.
4. One user per conversation only.

## Key Procedures

### Order Status Rules
- **pending**: can cancel or modify (address, payment, items)
- **processed**: cannot modify
- **delivered**: can return or exchange
- **cancelled**: no actions possible

### Cancel Pending Order
- Requires: order_id + reason ("no longer needed" OR "ordered by mistake" only).
- Refund timing: gift card = immediate, other = 5-7 business days.
- Always communicate the refund timeline to user.

### Modify Pending Order
- Can change: address, payment method, item variants (same product type only).
- Item modification can only be done ONCE per order.
- New payment must cover any price difference.
- Communicate all changes and any price differences to user.

### Return Delivered Order
- Collect: order_id, item_ids to return, payment method for refund.
- Refund must go to original payment OR existing gift card.
- Communicate the refund method and timing.

### Exchange Delivered Order
- Same product type only (e.g., shirt to shirt).
- Collect ALL items to exchange before proceeding.
- User pays/receives price difference.
- Communicate the price difference clearly.
"""

COMPRESSED_POLICY_AIRLINE = """# Airline Agent Policy

## Capabilities
Help users: book, modify, cancel flight reservations; handle refunds and compensation.

## Core Rules
1. Authenticate user first (by user_id, email, or name+DOB).
2. One tool call OR one response per turn, never both.
3. Confirm before any booking database update.
4. Transfer to human only if outside your capabilities.

## Key Procedures

### Booking Flights
- Collect: user_id, trip_type (one_way/round_trip), origin, destination, dates.
- Same cabin class across all flights in reservation.
- Max 5 passengers; collect name, DOB for each.
- Payment: max 1 travel certificate, 1 credit card, 3 gift cards (all from user profile).

### Modifying Reservations
- Look up all user reservations to find the correct one.
- Can change: flights, passengers, cabin class, baggage.
- Change fees depend on membership level.
- Basic economy has more restrictions.
- Communicate all fees and changes clearly.

### Cancellation & Refunds
- Verify the correct reservation before cancelling.
- Refund amount depends on: membership level, cabin class, timing.
- Travel certificates: non-refundable remainder.
- Gold members have maximum flexibility.
- Communicate the refund amount and method.

### Membership Levels
- regular: standard fees and restrictions
- silver: reduced fees, some flexibility
- gold: waived fees, maximum flexibility

### Cabin Classes
- basic_economy: most restricted
- economy: standard
- business: premium, most flexible
"""


def get_compressed_policy(domain: str) -> str:
    """Return compressed policy for domain, or empty string if not available."""
    policies = {
        "telecom": COMPRESSED_POLICY_TELECOM,
        "retail": COMPRESSED_POLICY_RETAIL,
        "airline": COMPRESSED_POLICY_AIRLINE,
    }
    return policies.get(domain, "")


def format_tools_json_schema(tools_openai: list[dict[str, Any]]) -> str:
    tool_schemas = []
    for tool in tools_openai:
        fn = tool.get("function", tool)
        schema = {
            "name": fn.get("name", "unknown"),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
        tool_schemas.append(schema)
    return json.dumps(tool_schemas, indent=2)


def build_tau2_agent_system_prompt(
    *,
    domain: str,
    policy: str,
    tools_openai: list[dict[str, Any]],
) -> str:
    if USE_COMPRESSED:
        compressed = get_compressed_policy(domain)
        if compressed:
            policy = compressed

    tools_text = format_tools_json_schema(tools_openai)
    return f"""## Output Format
Every turn: exactly ONE tool call in this format:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

Special actions:
- respond: <tool_call>{{"name": "respond", "arguments": {{"content": "message"}}}}</tool_call>
- done: <tool_call>{{"name": "done", "arguments": {{}}}}</tool_call>

---

You are a {domain} customer support agent. Complete the user's task following the policy below.

{policy}

## Available Tools
{tools_text}

## Rules
- One tool call per turn (no plain text responses)
- Authenticate user before state changes
- Confirm before modifications
- Communicate all relevant details to the user
- Call done immediately when task is complete
"""
