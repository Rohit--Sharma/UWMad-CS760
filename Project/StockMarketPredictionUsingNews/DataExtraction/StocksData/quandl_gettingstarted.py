# Install Quandl using "pip install quandl"
import quandl
import numpy

# A temporary API Key
quandl.ApiConfig.api_key = "HC3WjNA-trqxLy42S5Nm"

data = quandl.get("EIA/PET_RWTC_D", returns="numpy")
data = quandl.get("EOD/MSFT", returns="numpy")

data_table = quandl.get_table("MER/F1", paginate=True)
