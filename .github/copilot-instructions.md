Declarative functional code is preferred over imperative code.
Functions should be pure and avoid side effects, as long as performance is not significantly impacted.
It is easier to understand code that returns new data structures rather than mutating existing ones.
Do not be afraid of major redesigns if they lead to better code quality and maintainability. Minor code changes often lead to code plague and technical debt, while major redesigns can lead to cleaner and more robust code.
Do not be afraid of removing code that is no longer needed.
The become keyword is available now for tail calls, so you can write recursive functions without worrying about stack overflows.
Zero cost abstractions that make code safer and easier to read are encouraged. For example: type wrappers (often called newtypes) that prevent mixing up different types of data, or that provide additional functionality.
Code ordering should be F# style: nothing can be invoked or referenced before it is defined. This means that helper functions should be defined before the main function that uses them, and that the main function should be defined at the end of the file.
Always keep the public surface as simple and small as possible. If a function or type is only used internally, it should not be exposed publicly.
Often, the best change to make is to remove code.