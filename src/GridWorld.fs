module GridWorld.ReinforcementLearning

open Microsoft.FSharp.Reflection
// Environment

type ActionGridWorld =
  | Down
  | Right
  | Up
  | Left

type State = int * int

type Action = 
  | GridWorld of ActionGridWorld

type ActionNumber = int  

type Reward = float
type Done = bool

type Transitions = Map<State * Action, State * Reward * Done>

type StepFunction = State -> ActionNumber -> State * Reward * Done

//type Environment = Transitions
// possibly add continuous environments as well? Environment = Discrete | Continuous

let stepFunction (environment: Transitions) (state: State) (action: ActionNumber) =
  // index into the discriminated union of actions by the action index
  let cases = FSharpType.GetUnionCases typeof<ActionGridWorld>
  let a = FSharpValue.MakeUnion(cases.[action], [||]) :?> ActionGridWorld
  environment.[state, Action.GridWorld(a)]

let limit a b = if a < 0 then 0 else if a >= b then b-1 else a  

let simpleGridWorld() : StepFunction * State * int =
  let width, height = 3,3
  let terminal = (width-1, height-1)
  let transitions = 
    [ for x in 0..width-1 do 
        for y in 0..height-1 do
          yield ((x,y), Action.GridWorld Down),   ((x, limit (y-1) height), -1., false) 
          yield ((x,y), Action.GridWorld Right),  (( limit (x+1) width, y), -1., false) 
          yield ((x,y), Action.GridWorld Up),     ((x, limit (y+1) height), -1., false) 
          yield ((x,y), Action.GridWorld Left),   (( limit (x-1) width, y), -1., false) 
          ]  
    |> List.map (fun ((state, action), (state', reward, finished)) ->
        if state' = terminal then 
          ((state, action), (state', 100., true))
        else 
          ((state, action), (state', reward, finished)))
    |> Map
  let initialState = (0,0)
  let actionCount = 4
  let stateCount = width * height
  (stepFunction transitions),  initialState, actionCount

let cliffWalking() : StepFunction * State * int =
  let width = 12
  let height = 4
  let terminal = (width-1, 0)

  let transitions =
    [ for x in 0..width-1 do 
        for y in 0..height-1 do
          yield ((x,y), Action.GridWorld Down),   ((x, limit (y-1) height), -1., false) 
          yield ((x,y), Action.GridWorld Right),  (( limit (x+1) width, y), -1., false) 
          yield ((x,y), Action.GridWorld Up),     ((x, limit (y+1) height), -1., false) 
          yield ((x,y), Action.GridWorld Left),   (( limit (x-1) width, y), -1., false) 
          ]  
    |> List.map (fun ((state, action), (state', reward, finished)) ->
        if state' = terminal then 
          ((state, action), (state', 100., true))
        else 
          let x', y' = match state' with | (a,b) -> a,b
          if x' > 0 && x' < width-1 && y' = 0 then
            // step into the cliff area - penalty and back to begining
            (state, action), ((0,0), -100., false)
          else
            ((state, action), (state', reward, finished)))
    |> Map  
  let initialState = (0,0)
  let actionCount = 4
  (stepFunction transitions),  initialState, actionCount


//=================================


let stepFun, initialState, actionCount = simpleGridWorld()

let randomAgent (rnd: System.Random) step initialState actionCount =    

    let rec episode state totalReward steps =
        let action = rnd.Next(actionCount) // choose random action
        let state', reward, finished = step state action
        if finished then  
            totalReward + reward, steps + 1
        else
            episode state' (totalReward + reward) (steps + 1)

    episode initialState 0. 0

let randomPerformance nIter = 
    let rnd = System.Random(1)
    let step, initialState, actionCount = cliffWalking()// simpleGridWorld()
    for i in 1..nIter do
        let reward, stepsTaken = randomAgent rnd step initialState actionCount
        printfn "%d: %.0f reward, %d steps" i reward stepsTaken 

// -----------------------------------------------

type QLearningParameters = 
    { Epsilon: float; Alpha : float; Gamma: float }  

type ActionValueTable = System.Collections.Generic.Dictionary<State * ActionNumber, float>

let optimalPolicy (rnd: System.Random) actionCount state (actionValueTable: ActionValueTable) =
  let possibleActions = 
    [| for a in 0..actionCount-1 ->
        if not (actionValueTable.ContainsKey(state, a)) then 
            actionValueTable.Add((state, a), 0.0)
        a, actionValueTable.[state, a]
      |]
  let maxQ = possibleActions |> Array.maxBy snd |> snd
  possibleActions 
  |> Array.choose (fun (a, v) -> if abs(v - maxQ) < 1e-5 then Some a else None) 
  |> fun a -> a.[rnd.Next(a.Length)]
  
let getGreedyPolicy actionCount state (actionValueTable: ActionValueTable) =
  let possibleActions = 
    [ for a in 0..actionCount-1 ->
        if not (actionValueTable.ContainsKey(state, a)) then 
            actionValueTable.Add((state, a), 0.0)
        a, actionValueTable.[state, a]
      ]
  let maxQ = possibleActions |> List.maxBy snd |> snd

  let greedyActions =
    possibleActions 
    |> List.choose (fun (a, v) -> if abs(v - maxQ) < 1e-5 then Some a else None) 
 
  let cases = FSharpType.GetUnionCases typeof<ActionGridWorld>
  
  greedyActions
  |> List.map (fun actionIdx ->
    FSharpValue.MakeUnion(cases.[actionIdx], [||]) :?> ActionGridWorld
  )

let qLearningAgent (rnd: System.Random) step initialState actionCount actionValueTable parameters = 

  let rec episode state totalReward steps =

      let action =  // epsilon greedy policy
          if rnd.NextDouble() < parameters.Epsilon then
              rnd.Next(actionCount) // choose random action
          else
              optimalPolicy rnd actionCount state actionValueTable

      let state', reward, finished = step state action

      let argMaxAction = optimalPolicy rnd actionCount state' actionValueTable 

      let q' = 
        actionValueTable.[state, action] 
        + parameters.Alpha * (reward + parameters.Gamma * actionValueTable.[state', argMaxAction] - actionValueTable.[state, action])

      actionValueTable.[(state, action)] <- q'

      if finished then  
          actionValueTable, totalReward + reward, steps + 1
      else
          episode state' (totalReward + reward) (steps + 1)          

  episode initialState 0. 0


let sarsaAgent (rnd: System.Random) step initialState actionCount actionValueTable parameters = 

  let rec episode state action totalReward steps =
    let state', reward, finished = step state action
    let action' =  // epsilon greedy policy
        if rnd.NextDouble() < parameters.Epsilon then
            rnd.Next(actionCount) // choose random action
        else
            optimalPolicy rnd actionCount state' actionValueTable

    let q' = 
      if not (actionValueTable.ContainsKey(state, action)) then actionValueTable.Add((state, action), 0.)
      if not (actionValueTable.ContainsKey(state', action')) then actionValueTable.Add((state', action'), 0.)
      
      actionValueTable.[state, action] 
      + parameters.Alpha * (reward + parameters.Gamma * actionValueTable.[state', action'] - actionValueTable.[state, action])

    actionValueTable.[(state, action)] <- q'

    if finished then  
        actionValueTable, totalReward + reward, steps + 1
    else
        episode state' action' (totalReward + reward) (steps + 1)    

  let action =  // epsilon greedy policy
      if rnd.NextDouble() < parameters.Epsilon then
          rnd.Next(actionCount) // choose random action
      else
          optimalPolicy rnd actionCount initialState actionValueTable

  episode initialState action 0. 0



// Agent that always executes the action with the largest state-action value
let greedyPolicyAgent (rnd: System.Random) step initialState actionCount actionValueTable =

  let rec episode state totalReward steps =
      let action = optimalPolicy rnd actionCount state actionValueTable
      printfn "%A     %d" state action
      let state', reward, finished = step state action
      if finished then  
          totalReward + reward, steps + 1
      else
          episode state' (totalReward + reward) (steps + 1)

  episode initialState 0. 0
  

let evaluateAgent environment actionValueTable nIter = 
  let rnd = System.Random(1)
  let step, initialState, actionCount = environment()
          
  [ 1 .. nIter ]
  |> List.map (fun e -> 
      let reward, steps = greedyPolicyAgent rnd step initialState actionCount actionValueTable
      reward, steps)


let trainQlearningAgent environment nIter = 
  let rnd = System.Random(1)
  let step, initialState, actionCount = environment() // simpleGridWorld()

  let initialActionValueTable = ActionValueTable() 

  let parameters = {
      Epsilon = 0.1
      Alpha = 0.2
      Gamma = 0.7 }
          
  ((initialActionValueTable, [], []), [ 1 .. nIter ])
  ||> List.fold (fun (q, rws, nsteps) e -> 
      let q', reward, steps =  qLearningAgent rnd step initialState actionCount q parameters
      q', reward::rws, steps::nsteps)

let trainSarsaAgent environment nIter = 
  let rnd = System.Random(1)
  let step, initialState, actionCount = environment() // simpleGridWorld()

  let initialActionValueTable = ActionValueTable() 

  let parameters = {
      Epsilon = 0.1
      Alpha = 0.2
      Gamma = 0.7 }
          
  ((initialActionValueTable, [], []), [ 1 .. nIter ])
  ||> List.fold (fun (q, rws, nsteps) e -> 
      let q', reward, steps =  sarsaAgent rnd step initialState actionCount q parameters
      q', reward::rws, steps::nsteps)


// let actionValueTableQLearning, rewardsQLearning, stepsQLearning = trainQlearningAgent cliffWalking 100
// let resultsQLearning = evaluateAgent cliffWalking actionValueTableQLearning 10

// let actionValueTableSarsa, rewardsSarsa, stepsSarsa = trainSarsaAgent cliffWalking 500
// let resultsSarsa = evaluateAgent cliffWalking actionValueTableSarsa 10


// Ugly with dictionary!
// ToDo: version with probabilities & implement FrozenLake


