
def bar() :
  a = 10
  b = 2
  c = a * b
  return c

func foo() {
  return bar()
}

function main():
	return foo()


<: main() <: '\n'