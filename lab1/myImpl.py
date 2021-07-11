import util

"""
Data sturctures we will use are stack, queue and priority queue.

Stack: first in last out
Queue: first in first out
    collection.push(element): insert element
    element = collection.pop() get and remove element from collection

Priority queue:
    pq.update('eat', 2)
    pq.update('study', 1)
    pq.update('sleep', 3)
pq.pop() will return 'study' because it has highest priority 1.

"""

"""
problem is a object has 3 methods related to search state:

problem.getStartState()
Returns the start state for the search problem.

problem.isGoalState(state)
Returns True if and only if the state is a valid goal state.

problem.getChildren(state)
For a given state, this should return a list of tuples, (next_state,
step_cost), where 'next_state' is a child to the current state, 
and 'step_cost' is the incremental cost of expanding to that child.

"""
def myDepthFirstSearch(problem):
    # 栈实现的DFS
    visited = {}                # 存储搜索路径
    frontier = util.Stack()     # 存储当前搜索的边界节点

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            # state是目标状态，则逆向寻找路径
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]                
        
        if state not in visited:
            # 当前state没访问过，才做DFS
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myBreadthFirstSearch(problem):
    # YOUR CODE HERE
    # 队列实现的BFS
    visited = {}                # 存储搜索路径
    frontier = util.Queue()     # 存储当前搜索的边界节点

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            # state是目标状态，则逆向寻找路径
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]                
        
        if state not in visited:
            # 当前state没访问过，才做BFS
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myAStarSearch(problem, heuristic):
    # YOUR CODE HERE
    # 最小优先队列实现的A* search
    visited = {}                        # 存储搜索路径：state->prev_state
    frontier = util.PriorityQueue()     
    # 存储当前搜索的边界节点：[(state, prev_state, 初始状态到state的路径长度), state的评估值]
    # state的评估值 = 初始状态到state的路径长度 + state到目标状态的启发式估计值
    state = problem.getStartState()
    frontier.update((state, None, 0), heuristic(state))

    while not frontier.isEmpty():
        state, prev_state, length = frontier.pop()
        if problem.isGoalState(state):
            # state是目标状态，则逆向寻找路径
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            # 当前state没访问过，才做A* search
            visited[state] = prev_state
            # 将state的后继状态加入搜索边界，注意优先级是以对next_state的评估值
            for next_state, step_cost in problem.getChildren(state):
                frontier.update((next_state, state, length + step_cost), length + step_cost + heuristic(next_state))
    return []

"""
Game state has 4 methods we can use.

state.isTerminated()
Return True if the state is terminated. We should not continue to search if the state is terminated.

state.isMe()
Return True if it's time for the desired agent to take action. We should check this function to determine whether an agent should maximum or minimum the score.

state.getChildren()
Returns a list of legal state after an agent takes an action.

state.evaluateScore()
Return the score of the state. We should maximum the score for the desired agent.

"""
class MyMinimaxAgent():

    def __init__(self, depth):
        self.depth = depth

    def minimax(self, state, depth):

        if state.isTerminated() or depth==0:
            # state是终止状态或已经超过了搜索层数
            return None, state.evaluateScore()        

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')

        for child in state.getChildren():
            # 遍历state的子状态child
            if child.isMe():
                # child是目标agent操作
                _, child_score = self.minimax(child, depth-1)
            else:
                _, child_score = self.minimax(child, depth)
            if state.isMe() and child_score > best_score:
                # state是目标agent操作，则最大化自己的效用
                best_state = child
                best_score = child_score
                continue
            if state.isMe()==False and child_score < best_score:
                # state是目标agent的竞争者操作，则最小化对目标agent的效用
                best_state = child
                best_score = child_score
                
        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state

class MyAlphaBetaAgent():
    '''带alpha beta剪枝的minimax agent'''
    def __init__(self, depth):
        self.depth = depth

    def min_value(self, state, depth, alpha, beta):
        # state是目标agent的对手操作
        if state.isTerminated() or depth==0:
            # state是终止状态或已经超过了搜索层数
            return None, state.evaluateScore()   

        best_state, best_score = None, float('inf')
        alpha_new = alpha
        beta_new = beta

        for child in state.getChildren():
            # 遍历state的子状态child
            if child.isMe():
                # child是目标agent操作
                _, child_score = self.max_value(child, depth-1, alpha_new, beta_new)
            else:
                _, child_score = self.min_value(child, depth, alpha_new, beta_new)
            if child_score < best_score:
                # 最小化对目标agent的效用
                best_state = child
                best_score = child_score
            if child_score < alpha_new:
                # 剪枝
                return best_state, best_score
            beta_new = min(beta_new, child_score)
                
        return best_state, best_score
        

    def max_value(self, state, depth, alpha, beta):
        # state是目标agent操作
        if state.isTerminated() or depth==0:
            # state是终止状态或已经超过了搜索层数
            return None, state.evaluateScore()   

        best_state, best_score = None, -float('inf')
        alpha_new = alpha
        beta_new = beta

        for child in state.getChildren():
            # 遍历state的子状态child
            if child.isMe():
                # child是目标agent操作
                _, child_score = self.max_value(child, depth-1, alpha_new, beta_new)
            else:
                _, child_score = self.min_value(child, depth, alpha_new, beta_new)
            if child_score > best_score:
                # 最大化自己的效用
                best_state = child
                best_score = child_score
            if child_score > beta_new:
                # 剪枝
                return best_state, best_score
            alpha_new = max(alpha_new, child_score)
                
        return best_state, best_score


    def getNextState(self, state):
        # 这里保证根节点和第一层必然不会被剪去，故初始beta必须为正无穷
        best_state, _ = self.max_value(state, self.depth, -float('inf'), float('inf'))
        return best_state
