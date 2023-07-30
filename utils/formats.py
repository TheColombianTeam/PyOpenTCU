from utils.args import args
from typing import _GenericAlias
from sfpy.posit import Posit16, Posit32
from sfpy.float import Float16, Float32

type_name: str = args.config.format.lower()
TOTALBITS: int = int(type_name[-2:])

if type_name == "posit16":
    CustomData: _GenericAlias = Posit16
elif type_name == "posit32":
    CustomData: _GenericAlias = Posit32
elif type_name == "float16":
    CustomData: _GenericAlias = Float16
else:
    CustomData: _GenericAlias = Float32
