import python;

trace("HELLO");

python("print('PYTHON WORKS')");

python("""
import inspect, sys
print(inspect.getfile(inspect))
sys.stdout.flush()
""");
