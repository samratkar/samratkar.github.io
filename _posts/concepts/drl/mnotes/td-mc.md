Monte Carlo update for action-value function Q(s,a):

$[ Q(s,a) \leftarrow Q(s,a) + \alpha \bigl(G_t - Q(s,a)\bigr) ]$
where:

$( \alpha )$ is the learning rate
$( G_t )$ is the total return (sum of rewards) following the episode starting from state ( s ) and action ( a )

TD SARSA update for Q(s,a):
$[ Q(s,a) \leftarrow Q(s,a) + \alpha \bigl(r + \gamma Q(s',a') - Q(s,a)\bigr) ]$
where:

$( r )$ is the immediate reward after taking action ( a ) in state ( s )
$( \gamma )$ is the discount factor
$( s' )$ and $( a' )$ are the next state and next action
