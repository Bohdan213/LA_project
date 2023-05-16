def converter(input_data):
    """
    # input_data structure: [([x1, y1, h1, w1], ""), ([x2, y2, h2, w2], ""), ...]
    # input_data[i][0]: choords
    # input_data[i][1]: symbol
    # it converts input_data to latex code
    # using choords it determines where to put the symbols
    # output: latex code
    """
    fixed_data = [(_[0][1]+_[0][2]/3, _[0][1]+_[0][2]/3*2, _[1])
                  for _ in input_data]
    signs = ["+", "-", "*", "/"]
    special = {"belongs":"\in ", "big_e":"\exists ", "big_a":"\\forall "}

    prev_elem = fixed_data[0]
    output = fixed_data[0][2] if fixed_data[0][2] not in special else special[fixed_data[0][2]]

    for elem, old in zip(fixed_data[1:], input_data[1:]):
        if not elem[2] in signs and not prev_elem[2] in signs:
            if elem[0] > prev_elem[1]:
                if old[0][1] > prev_elem[0]:
                    output += "_"
            elif elem[1] < prev_elem[0]:
                if old[0][1]+old[0][2] < prev_elem[1]:
                    output += "^"
        if elem[2] in special:
            output += special[elem[2]]
        else:
            output += elem[2]
        prev_elem = elem
    return output


if __name__ == "__main__":
    # test
    print(converter([((165, 149, 193, 104), '2'), ((399, 52, 251, 109), 'b'), ((
        603, 180, 126, 101), '+'), ((813, 159, 161, 91), '6'), ((998, 163, 134, 117), 'a')]))
