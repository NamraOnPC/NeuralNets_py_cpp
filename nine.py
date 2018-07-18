from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T

random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1


print "training_set_inputs"
print training_set_inputs
print "synaptic_weights"
print synaptic_weights
print "dot(training_set_inputs, synaptic_weights)"
print dot(training_set_inputs, synaptic_weights)

print "output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))"
output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
print "output"
print output

for iteration in xrange(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print "1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))"
print 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))


'''
training_set_inputs
[[0 0 1]
 [1 1 1]
 [1 0 1]
 [0 1 1]]
synaptic_weights
[[-0.16595599]
 [ 0.44064899]
 [-0.99977125]]
dot(training_set_inputs, synaptic_weights)
[[-0.99977125]
 [-0.72507825]
 [-1.16572724]
 [-0.55912226]]
output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
output
[[ 0.2689864 ]
 [ 0.3262757 ]
 [ 0.23762817]
 [ 0.36375058]]
1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))
[ 0.99993704]
'''
