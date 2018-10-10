"""
decision tree
"""

using BenchmarkTools

import CSV
import DataFrames
import Random
import Plots

"""
	Leaf

Define the leaf of the decision tree, `Leaf` contains the label.
"""
struct Leaf
    label::Int
end

"""
	Node

Define the node of the decision tree, `Node` contains the attribute it chooses.
"""
struct Node
    attribute::Int
    son::Vector{Union{Leaf, Node}}
end

function showTree(io::IO, t::Leaf, dep::Int, str::String = "")
	println(io, "\t"^dep * str * "Leaf: label: $(t.label)")
end

function showTree(io::IO, t::Node, dep::Int, str::String = "")
	println(io, "\t"^dep * str * "Node: attribute: $(t.attribute)")
	for (i, son) in enumerate(t.son)
		showTree(io, son, dep + 1, "son$i ")
	end
end

# dump(Node(1, [Node(2, [Leaf(1), Leaf(2)]), Leaf(2)]))

# v should be Array{Int, 2}, every attribute should be Int
"""
	getLabel(t::Union{Leaf, Node}, v)

`getLabel` is the function to get one vector's label, it goes from the root to
the Leaf and return the leaf's label.
"""
getLabel(t::Leaf, v) = t.label
getLabel(t::Node, v) = getLabel(t.son[v[t.attribute]], v)

"""
	splitData(data, attribute, numOfClasses)

It is the function to split `data` according to the `attribute`.
`data` is the vector of vectors(examples),
`attribute` is the chosen attribute's id, it is an integer,
`numOfClasses` is the chosen attribute's classes.
"""
function splitData(data, attribute::Int, numOfClasses::Int)
    map(1:numOfClasses) do i
        collect(v for v in data if v[attribute] == i)
    end
end

#=
splitData([ [3, 5], [2, 3]], 1, 3)

data = [[2, 1], [3,1], [4, 1], [5, 2]]
all(v[end] == data[1][end] for v in data)
=#

"""
	build(data, attributes, findAttribute, labelSize)

It is the function to build the decision tree.
`data` is the vector of vectors(examples),
`attributes` is the set of attributes' id which can be used,
`findAttribute` is the function to choose the attribute, it depends on the
model you choose,
`labelSize` is the vector of number of attributes' classes.
"""
function build(
    data::Vector{Vector{Int}},
    attributes::Set{Int},
    findAttribute::Function,
    labelSize::Vector{Int}
    )::Union{Leaf, Node}

	# Here are three kinds of bound situations.
    if isempty(data)
        return Leaf(0)
    elseif isempty(attributes)
        sumup = zeros(Int, labelSize[end])
        for v in data
        	sumup[v[end]] += 1
        end
        return Leaf(findmax(sumup)[2])
    elseif all(v[end] == data[1][end] for v in data)
        return Leaf(data[1][end])
    end

    attr = findAttribute(data, attributes, labelSize)
    pop!(attributes, attr)
    datas = splitData(data, attr, labelSize[attr])
    sons = map(x -> build(x, attributes, findAttribute, labelSize), datas)
    push!(attributes, attr)
    Node(attr, sons)
end


"""
	countLabel(data, attr, num)

It is the function to split `data` according to attribute `attr` and count
the number of every classes.
The result is the frequency of every classes.
"""
function countLabel(data, attr::Int, num::Int)
    if size(data)[1] == 0
        return zeros(num)
    end
    sumup = zeros(Int, num)
    for v in data
        sumup[v[attr]] += 1
    end
    sumup ./ length(data)
end

#=
data = [[1, 2, 3], [2, 1, 1], [2, 1, 1], [2, 3, 2]]
countLabel(data, 3, 3)
=#

"""
	findFunction_ID3(data, attributes, labelSize)

It is the function to choose the attribute from the set `attributes`.
"""
function findFunction_ID3(data, attributes, labelSize)::Int
    H(p) = iszero(p) ? 0.0 : -p * log2(p)
    m = length(labelSize)

    H_D = sum(H, countLabel(data, m, labelSize[m])) # H(D)
    H_D_A = map(attr for attr in attributes) do attr
        datas = splitData(data, attr, labelSize[attr])
        Hs = map(datas) do D
            sum(H, countLabel(D, m, labelSize[m]))
        end
        Ps = countLabel(data, attr, labelSize[attr])
        sum(@. Hs * Ps), attr
    end
    minimum(H_D_A)[2]
end

"""
	findFunction_C45(data, attributes, labelSize)

It is the function to choose the attribute from the set `attributes`.
"""
function findFunction_C45(data, attributes, labelSize)::Int
    H(p) = iszero(p) ? 0.0 : -p * log2(p)
    m = length(labelSize)

    H_D = sum(H, countLabel(data, m, labelSize[m])) # H(D)
    H_A   = map(attr for attr in attributes) do attr
        sum(H, countLabel(data, attr, labelSize[attr]))
    end
    H_D_A = map(attr for attr in attributes) do attr
        datas = splitData(data, attr, labelSize[attr])
        Hs = map(datas) do D
            sum(H, countLabel(D, m, labelSize[m]))
        end
        Ps = countLabel(data, attr, labelSize[attr])
        sum(@. Hs * Ps), attr
    end
    gain_ratio = map((x, y) -> ((H_D - y[1]) / x, y[2]), H_A, H_D_A)
    maximum(gain_ratio)[2]
end

"""
	findFunction_CART(data, attributes, labelSize)

It is the function to choose the attribute from the set `attributes`.
"""
function findFunction_CART(data, attributes, labelSize)::Int
	gini(v) = sum(p -> p * (1 - p), v)
	m = length(labelSize)

	g_D = gini(countLabel(data, m, labelSize[m]))
	g_D_A = map(attr for attr in attributes) do attr
		datas = splitData(data, attr, labelSize[attr])
		Ps = countLabel(data, attr, labelSize[attr])
		Gs = map(datas) do D
			gini(countLabel(D, m, labelSize[m]))
		end
		sum(@. Gs * Ps), attr
	end
	minimum(g_D_A)[2]
end

train_set = CSV.read(pwd() * "/data/Car_train.csv")

"""
	getDict(dataset)

It is the function to get the dictionaries which map the attributes to integer.
"""
function getDict(dataset)
    map(1:7) do j
        T = typeof(dataset[j][1])
        dict = Dict{T, Int}()
        numOfClasses = 0
        for w in dataset[j]
            if !haskey(dict, w)
                numOfClasses += 1
                dict[w] = numOfClasses
            end
        end
        dict
    end
end

# getDict(train_set)

"""
	transformDataSet(dataset, dictionaries)

It is the function to transform the dataset to vectors of vectors(examples)
	of integers according to dictionaries.
"""
function transformDataSet(dataset, dictionaries)
    ys = map(1:7) do j
        dict = dictionaries[j]
        map(dataset[j]) do w
            haskey(dict, w) ? dict[w] : -1
        end
    end
    (n, ) = size(ys[1])
    xs = [[ys[j][i] for j = 1:7] for i = 1:n]
end

# transformDataSet(train_set, getDict(train_set))

"""
	k_fold_cross_validation(K, dataset, dictionaries, labelSize, findFunction)

It is the function to get the accuracy of the model.
"""
function k_fold_cross_validation(
    K::Int,
    dataset, dictionaries, labelSize,
    findFunction
    )
    Random.shuffle!(dataset)
    n = length(dataset)
    m = length(dictionaries)
    blockSize = floor(Int, n / K)
    map(1:K) do i
        valid_set = dataset[1 + (i - 1) * blockSize : i * blockSize]
        train_set = Vector{Vector{Int}}(undef, n - blockSize)
        train_set[1 : (i - 1) * blockSize] = dataset[1 : (i - 1) * blockSize]
        train_set[1 + (i - 1) * blockSize : n - blockSize] = dataset[1 + i * blockSize : n]

        tree = build(train_set, Set(1:(m - 1)), findFunction, labelSize)
        predict = map(v -> getLabel(tree, v), valid_set)
        numOfRight = sum(zip(predict, valid_set)) do (p, v)
            p == v[m]
        end
        numOfRight / blockSize
    end |> sum |> (x -> x / K)
end


dictionaries = getDict(train_set)
labelSize = map(d -> length(d), dictionaries)
dataset = transformDataSet(train_set, dictionaries)

# show the three kinds of trees
begin
	tree1 = build(dataset, Set(1:length(labelSize) - 1), findFunction_ID3, labelSize)
	open("ID3_Tree.txt", "w") do f
		showTree(f, tree1, 0, "")
	end
	tree2 = build(dataset, Set(1:length(labelSize) - 1), findFunction_C45, labelSize)
	open("C45_Tree.txt", "w") do f
		showTree(f, tree2, 0, "")
	end
	tree3 = build(dataset, Set(1:length(labelSize) - 1), findFunction_CART, labelSize)
	open("CART_Tree.txt", "w") do f
		showTree(f, tree3, 0, "")
	end
end

# choss the best model
begin
	line_K = 2:5
	line_ID3 = zeros(Float64, length(line_K))
	foreach(1:10) do _
		line_ID3 .+= map(line_K) do k
			k_fold_cross_validation(k, dataset, dictionaries, labelSize, findFunction_ID3)
		end
	end
	line_ID3 = line_ID3 ./ 10
	line_C45 = zeros(Float64, length(line_K))
	foreach(1:10) do _
		line_C45 .+= map(line_K) do k
			k_fold_cross_validation(k, dataset, dictionaries, labelSize, findFunction_C45)
		end
	end
	line_C45 = line_C45 ./ 10
	line_CART = zeros(Float64, length(line_K))
	foreach(1:10) do _
		line_CART .+= map(line_K) do k
			k_fold_cross_validation(k, dataset, dictionaries, labelSize, findFunction_CART)
		end
	end
	line_CART = line_CART ./ 10
	figure = Plots.plot(line_K, [line_ID3, line_C45, line_CART])
	Plots.savefig(figure, "model_compare.png")
end


# label the test set
begin
	test_set = CSV.read(pwd() * "/data/Car_test.csv")
	m = length(dictionaries)
	tree = build(transformDataSet(train_set, dictionaries), Set(1:(m - 1)), findFunction_C45, labelSize)
	predict = map(v -> getLabel(tree, v), transformDataSet(test_set, dictionaries))
	reverseDict = collect(keys(dictionaries[m]))
	result = map(predict) do v
		iszero(v) ? rand(reverseDict) : reverseDict[v]
	end
	# @show reverseDict
	df = DataFrames.DataFrame(label = result)
	CSV.write("result.csv", df)
end

#=
data = [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 2, 1],
    [2, 1, 1, 1, 2],
    [3, 2, 1, 1, 2],
    [3, 3, 2, 1, 2],
    [3, 3, 2, 2, 1],
    [2, 3, 2, 2, 2],
    [1, 2, 1, 1, 1],
    [1, 3, 2, 1, 2],
    [3, 2, 2, 1, 2],
    [1, 2, 2, 2, 2],
    [2, 2, 1, 2, 2],
    [2, 1, 2, 1, 2],
    [3, 2, 1, 2, 1]]
labelSize = [3, 3, 2, 2, 2]

findFunction_CART(data, Set([1, 2, 3, 4]), labelSize)
build(data, Set([1, 2, 3, 4]), findFunction_CART, labelSize)
build(data, Set([1, 2, 3, 4]), findFunction_ID3, labelSize)
=#

#=
# 年龄 收入 学生 信用
data = [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 1],
    [1, 2, 1, 1, 1, 2],
    [2, 3, 2, 1, 1, 2],
    [2, 3, 3, 2, 1, 2],
    [2, 3, 3, 2, 2, 1],
    [3, 2, 3, 2, 2, 2],
    [3, 1, 2, 1, 1, 1],
    [6, 1, 3, 2, 1, 2],
    [4, 3, 2, 2, 1, 2],
    [4, 1, 2, 2, 2, 2],
    [5, 2, 2, 1, 2, 2],
    [5, 2, 1, 2, 1, 2],
    [6, 3, 2, 1, 2, 1]]
labelSize = [6, 3, 3, 2, 2, 2]
findFunction_ID3(data, Set([1, 2, 3, 4, 5]), labelSize)
findFunction_C45(data, Set([1, 2, 3, 4, 5]), labelSize)
tree1 = build(data, Set([1, 2, 3, 4, 5]), findFunction_ID3, labelSize)
tree2 = build(data, Set([1, 2, 3, 4, 5]), findFunction_C45, labelSize)
dump(tree1)

dump(tree2)

getLabel(root, [1, 3, 1,1])
=#


#=
test_data = [
    [1, 2, 1],
    [2, 1, 2],
    [3, 2, 1]]
findFunction_ID3(test_data, Set([1, 2]), [3, 2, 2])
findFunction_C45(test_data, Set([1, 2]), [3, 2, 2])
=#
