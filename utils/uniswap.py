import json
import os

"""
Gloabl State
"""

path = os.path.dirname(os.path.abspath(__file__)) + '/..' + '/config' + '/config.json'

with open(path, "r") as f:
    config = json.load(f)

pool_address = config["pool_address"]
chain = config["chain"]
token0 = config["token0"]
token1 = config["token1"]
decimal_0 = int(config["decimal_0"])
decimal_1 = int(config["decimal_1"])
fee_tier = float(config["fee_tier"])
tickspacing = int(config["tickspacing"])

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

def tick_2_unreadable_price(tick):
    """
    tick: tick number, (int)
    return: price, (float)
    """
    return float(1.0001 ** tick)

def tick_2_readable_price(tick):
    """
    tick: tick number, (int)
    return: price, (float)
    """
    return float(1.0001 ** tick * 10 ** (decimal_0-decimal_1))

def tick_2_tickspacing(tick):
    """
    tick: tick number, (int)
    return: tick in tickspacing, (int)
    """
    return int(tick - tick%tickspacing)

def transform_price_2_unreadable(price_readable):
    """
    price_readable: price of the pool, (float)
    return: price, (float)
    """

    return float(price_readable * (10 ** (decimal_1-decimal_0)))

def transform_price_2_readable(price_unreadable):
    """
    price_unreadable: price of the pool, (float)
    return: price, (float)
    """

    return float(price_unreadable * (10 ** (decimal_0-decimal_1)))

def transform_x_amount_2_unreadable(x_amount_readable):
    """
    x_amount_readable: amount of token x, (float)
    return: x_amount, (int)
    """
    return int(x_amount_readable * (10 ** (decimal_0)))

def transform_x_amount_2_readable(x_amount_unreadable):
    """
    x_amount_unreadable: amount of token x, (float)
    return: x_amount, (float)
    """
    return float(x_amount_unreadable * (10 ** (-decimal_0)))

def transform_y_amount_2_unreadable(y_amount_readable):
    """
    y_amount_readable: amount of token y, (float)
    return: y_amount, (int)
    """
    return int(y_amount_readable * (10 ** (decimal_1)))

def transform_y_amount_2_readable(y_amount_unreadable):
    """
    y_amount_unreadable: amount of token y, (float)
    return: y_amount, (float)
    """
    return float(y_amount_unreadable * (10 ** (-decimal_1)))

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

def x_amount(price, L, R):
    """
    price: price of the pool, (float)
    L: liquidity, (int)
    R: range, (touple), R=(a,b), a,b are the tick numbers(int), assume a<b
    return: amount of token x, (float)
    """

    # case 1: Price above the range
    if price >= (tick_2_readable_price(R[1])):
      return float(0)

    # case 2: Price below the range
    elif price <= (tick_2_readable_price(R[0])):
      return transform_x_amount_2_readable( L * ( (tick_2_unreadable_price(R[1]/2)) - (tick_2_unreadable_price(R[0]/2)) ) / ( (tick_2_unreadable_price(R[1]/2)) * (tick_2_unreadable_price(R[0]/2)) ) )

    # case 3: Price within the range
    else:
      return transform_x_amount_2_readable( L * ( (tick_2_unreadable_price(R[1]/2)) - ((transform_price_2_unreadable(price))**(1/2)) ) / ( (tick_2_unreadable_price(R[1]/2)) * ((transform_price_2_unreadable(price))**(1/2)) ) )

def y_amount(price, L, R):
    """
    price: price of the pool, (float)
    L: liquidity, (int)
    R: range, (touple), R=(a,b), a,b are the tick numbers(int), assume a<b
    return: amount of token y, (float)
    """

    # case 1: Price above the range
    if price >= (tick_2_readable_price(R[1])):
      return transform_y_amount_2_readable( L * ( (tick_2_unreadable_price(R[1]/2)) - (tick_2_unreadable_price(R[0]/2)) ) )

    # case 2: Price below the range
    elif price <= (tick_2_readable_price(R[0])):
      return float(0)

    # case 3: Price within the range
    else:
      return transform_y_amount_2_readable( L * ( ((transform_price_2_unreadable(price))**(1/2)) - (tick_2_unreadable_price(R[0]/2)) ) )
    
def delta_x_amount(price_in, price_out, L):
    """
    price_in: price 1, (float)
    price_out: price 2, (float)
    L: liquidity, (int)
    return: delta amount of token x with hedging, (float)
    """

    return transform_x_amount_2_readable( L * ( (transform_price_2_unreadable(price_in)**(1/2))-(transform_price_2_unreadable(price_out)**(1/2)) ) / ( (transform_price_2_unreadable(price_in)**(1/2))*(transform_price_2_unreadable(price_out)**(1/2)) ) )
   
def delta_y_amount(price_in, price_out, L):
    """
    price_in: price 1, (float)
    price_out: price 2, (float)
    L: liquidity, (int)
    return: delta amount of token y with hedging, (float)
    """

    return transform_y_amount_2_readable( L * ( (transform_price_2_unreadable(price_out)**(1/2))-(transform_price_2_unreadable(price_in)**(1/2)) ) )
    
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
    
def Liquidity_from_x_amount(x_amount, R):
    """
    x_amount: amount of token x, (float)
    R: range, (touple), R=(a,b), a,b are the tick numbers(int), assume a<b
    return: Liquidty, (int)
    """

    return int(transform_x_amount_2_unreadable(x_amount) * ((tick_2_unreadable_price(R[1])**(1/2))*(tick_2_unreadable_price(R[0])**(1/2))) / ((tick_2_unreadable_price(R[1])**(1/2)) -(tick_2_unreadable_price(R[0])**(1/2))))

def Liquidity_from_y_amount(y_amount, R):
    """
    y_amount: amount of token y, (float)
    R: range, (touple), R=(a,b), a,b are the tick numbers(int), assume a<b
    return: Liquidty, (int)
    """

    return int(transform_y_amount_2_unreadable(y_amount) / ((tick_2_unreadable_price(R[1])**(1/2)) -(tick_2_unreadable_price(R[0])**(1/2))))

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

def swap_fee(close_price, x_scaled_vol, y_scaled_vol, L):
    """
    close_price: close price of the cycle, (float)
    x_scaled_vol: scaled volume of token x in a cycle, (float)
    x_scaled_vol: scaled volume of token y in a cycle, (float)
    L: liquidity, (int)
    return: swap fee, (touple): swap fee = (swap_fee_x, swap_fee_y, swap_fee_total), (float, float, float)
    """

    swap_fee_x = x_scaled_vol*L*(fee_tier/(1-fee_tier))
    swap_fee_y = y_scaled_vol*L*(fee_tier/(1-fee_tier))
    swap_fee_total = close_price*swap_fee_x + swap_fee_y

    return (float(swap_fee_x), float(swap_fee_y), float(swap_fee_total))

def LVR(price_in, price_out, L):
    """
    price_in: price of position created in a cycle, (float)
    price_out: price of position closed in a cycle, (float)
    L: liquidity, (float)
    return: LVR (float)
    """
    
    # print("="*100)
    # print("delta x amount")
    # print(delta_x_amount(price_in, price_out, L))
    # print("="*100)
    # print("delta y amount")
    # print(delta_y_amount(price_in, price_out, L))
    # print("="*100)

    return float( price_out * delta_x_amount(price_in, price_out, L) + delta_y_amount(price_in, price_out, L))

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

def liquidity_multiplier(t_0, k):
    """
    t_0: current price in tick, (int)
    k: width of range, (int)
    return: unit transformation, (float)
    """

    return float(1.0001 ** ((2 * t_0 + k * tickspacing) / 2)) / (2 * (1.0001 ** ((3 * t_0 + k * tickspacing) / 2) - 1.0001 ** ((3 * t_0) / 2)))