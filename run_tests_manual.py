import sys
import os
import traceback

# Add root to sys.path
sys.path.append(os.getcwd())

# Import tests directly now that path is set
try:
    print("=== Running test_guardrails_restructure ===")
    import tests.test_guardrails_restructure as t_restruct
    t_restruct.test_restructure_success()
    t_restruct.test_restructure_fail_pti()
    t_restruct.test_restructure_fail_dscr()
    print("✅ test_guardrails_restructure PASSED")
except Exception:
    traceback.print_exc()

print("\n")

try:
    print("=== Running test_guardrails_sell ===")
    import tests.test_guardrails_sell as t_sell
    t_sell.test_sell_success()
    t_sell.test_sell_fail_fire_sale_blocked()
    t_sell.test_sell_fail_price_negative()
    print("✅ test_guardrails_sell PASSED")
except Exception:
    traceback.print_exc()
