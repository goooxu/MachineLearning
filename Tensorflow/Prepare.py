import numpy as np

def random_expressions(lower_bound, upper_bound, operators, max_op_count, size):
    
    op_counts = [int(count ** 0.25) for count in np.random.randint((max_op_count + 1) ** 4, size=size)]
    nums = np.random.randint(lower_bound, upper_bound, size=sum(op_counts) + size)
    ops = np.random.randint(len(operators), size=sum(op_counts))

    for op_count in op_counts:
        elements = [nums[0]]
        for j in range(op_count):
            elements += [operators[ops[j]], nums[j + 1]]
        
        in_exp = ''.join([str(element) for element in elements])
        out_exp = str(eval(in_exp))
        yield (in_exp, out_exp)

        nums = nums[op_count + 1:]
        ops = ops[op_count:]

def print_expressions(expressions):
    for expression in expressions:
        print('{},{}'.format(expression[0], expression[1])) 

if __name__ == '__main__':
    operators = ['+','-','*']
    print_expressions(random_expressions(0, 10, operators, 3, 3900))
    print_expressions(random_expressions(0, 100, operators, 3, 58600))
    print_expressions(random_expressions(0, 1000, operators, 3, 253900))
    print_expressions(random_expressions(0, 10000, operators, 3, 683600))