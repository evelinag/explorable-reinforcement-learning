module App

(**
 The famous Increment/Decrement ported from Elm.
 You can find more info about Elmish architecture and samples at https://elmish.github.io/
*)

open Elmish
open Elmish.React
open Fable.Helpers.React
open Fable.Helpers.React.Props

open GridWorld.ReinforcementLearning

// MODEL

let cellSize = 40
let cellSpacing = 5
let width = 12
let height = 4
let rnd = System.Random()

module Colours =
  let positionColour = "#ff2e63"
  let positionHighlight = "#e10039"
  let gridColour = "#eaeaea"
  let cliffColour = "#696464"
  let startColour = "#08d9d6"
  let endColour = "#08d96e"
  let policyColour = "#423f3f"

type ReinforcementLearningAlgorithm = 
  | Random
  | QLearning
  | Sarsa
  | SarsaLambda

type ReinforcementLearningTask = {
    Step : (State -> ActionNumber -> State * Reward * Done)
    InitialState : State
    TerminalState : State
    Cliff : State []
    ActionCount : int
}

type Model = {
  Algorithm : ReinforcementLearningAlgorithm
  Task : ReinforcementLearningTask
  Grid : int []
  InitialPosition : State
  Position : State
  PathTaken : State list
  ActionValueTable : ActionValueTable
  TotalReward : float
  Steps : int
  Finished : bool
  Parameters : QLearningParameters
  ShowGreedyPolicy : bool
}


type Msg =
  | Step
  | ShowPolicy
  | Reset
  | RunEpisode
  | NewEpisode
  | SetAlgorithm of ReinforcementLearningAlgorithm


let init() = 
  let step, initialState, actionCount = cliffWalking()
  {
    Task = {
      Step = step
      InitialState = initialState 
      TerminalState = (11, 0)
      ActionCount = actionCount
      Cliff = [| for x in 1 .. 10 -> x, 0 |]  // This is for visualisation purposes - not part of the model
    }
    Grid = [||]
    InitialPosition = initialState
    Position = initialState
    ActionValueTable = ActionValueTable()
    TotalReward = 0.
    Steps = 0
    Algorithm = Random // QLearning //Random
    Finished = false
    Parameters = {
      Epsilon = 0.1
      Alpha = 0.2
      Gamma = 0.7 }
    ShowGreedyPolicy = false
    PathTaken = [initialState]
  }, Cmd.none


// UPDATE

let update (msg:Msg) (model:Model) =
  match msg with
  | Step -> 
      if model.Position = model.Task.TerminalState then
        model, Cmd.ofMsg NewEpisode 
      else
      
      let model', cmd = 
        match model.Algorithm with
        | Random ->
          let action = rnd.Next(model.Task.ActionCount) // choose random action
          let state', reward, finished = model.Task.Step model.Position action
          { model with 
              Finished = finished
              Position = state'; 
              TotalReward = model.TotalReward + reward
              Steps = model.Steps + 1;
              PathTaken = state' :: model.PathTaken }, Cmd.none
        
        | QLearning | Sarsa | SarsaLambda ->  
          // epsilon-greedy step
            let action =  // epsilon greedy policy
              if rnd.NextDouble() < model.Parameters.Epsilon then
                  rnd.Next(model.Task.ActionCount) // choose random action
              else
                  optimalPolicy rnd model.Task.ActionCount model.Position model.ActionValueTable
            
            let state', reward, finished = model.Task.Step model.Position action
            { model with 
                Finished = finished
                Position = state'; 
                TotalReward = model.TotalReward + reward
                Steps = model.Steps + 1
                PathTaken = state' :: model.PathTaken }, Cmd.none

      if model'.Position = model.InitialPosition then
        { model' with PathTaken = [ model.InitialPosition] }, cmd
      else
        model', cmd  

  | ShowPolicy -> 
      { model with ShowGreedyPolicy = if model.ShowGreedyPolicy then false else true }, Cmd.none

  | RunEpisode ->
      match model.Algorithm with
      | Random ->
        let reward, stepsTaken = randomAgent rnd model.Task.Step model.Task.InitialState model.Task.ActionCount
        { model with 
            Finished = true;
            TotalReward = reward; 
            Steps = stepsTaken
            }, Cmd.none
      
      | QLearning ->
        let actionValueTable, reward, steps =  
            qLearningAgent rnd model.Task.Step model.Task.InitialState model.Task.ActionCount model.ActionValueTable model.Parameters
        { model with 
            Finished = true;
            TotalReward = reward; 
            Steps = steps
            ActionValueTable = actionValueTable
            }, Cmd.none

      | Sarsa ->
        let actionValueTable, reward, steps =  
            sarsaAgent rnd model.Task.Step model.Task.InitialState model.Task.ActionCount model.ActionValueTable model.Parameters
        { model with 
            Finished = true;
            TotalReward = reward; 
            Steps = steps
            ActionValueTable = actionValueTable
            }, Cmd.none      

  | NewEpisode ->
      { model with
          Position = model.InitialPosition
          TotalReward = 0.
          Steps = 0
          Finished = false
          PathTaken = [ model.InitialPosition ]
      } , Cmd.none            

  | Reset -> init()

  | SetAlgorithm a ->
      let model', _ =  init()
      { model' with Algorithm = a }, Cmd.none

// VIEW (rendered with React)

let viewArrow x y (direction: ActionGridWorld) =
  let arrowLength = 10.
  let arrowWidth = 10.

  // coordinates of the top left corner of the cell
  let xCell = float (cellSize * x) 
  let yCell = float (cellSize * (height - y - 1)) 

  let pathD = 
    match direction with
    | Right ->
      let xLoc = xCell + float cellSize/2. - float cellSpacing/2. + arrowWidth/2.
      let yLoc = yCell + float cellSize/2.0 - arrowLength/2. - float cellSpacing/2.
      (sprintf "M%f,%f L%f,%f L%f,%f z" xLoc yLoc xLoc (yLoc + arrowWidth) (xLoc + arrowLength) (yLoc + arrowWidth/2.)) 
    | Down ->
      let xLoc = xCell + float cellSize/2. - float cellSpacing/2. - arrowWidth/2.
      let yLoc = yCell + float cellSize/2.0 - arrowLength/2. - float cellSpacing/2. + arrowWidth
      (sprintf "M%f,%f L%f,%f L%f,%f z" xLoc yLoc (xLoc + arrowWidth) yLoc (xLoc + arrowWidth/2.) (yLoc + arrowLength)) 
    | Up ->
      let xLoc = xCell + float cellSize/2. - float cellSpacing/2. - arrowWidth/2.
      let yLoc = yCell + float cellSize/2.0 - arrowLength/2. - float cellSpacing/2. 
      (sprintf "M%f,%f L%f,%f L%f,%f z" xLoc yLoc (xLoc + arrowWidth) yLoc (xLoc + arrowWidth/2.) (yLoc - arrowLength))
    | Left ->
      let xLoc = xCell + float cellSize/2. - float cellSpacing/2. - arrowWidth/2.
      let yLoc = yCell + float cellSize/2.0 - arrowLength/2. - float cellSpacing/2.
      (sprintf "M%f,%f L%f,%f L%f,%f z" xLoc yLoc xLoc (yLoc + arrowWidth) (xLoc - arrowLength) (yLoc + arrowWidth/2.)) 

  path [
    D pathD
    SVGAttr.Fill Colours.policyColour 
  ] []

let viewPosition x y colour = 
  let r = float (cellSize - 5*cellSpacing)/2.  // radius of the point
  let r' = float (cellSize - cellSpacing)/2.   // position offset of the centre
  
  let x' = float cellSize * float x + r'
  let y' = float (cellSize * (height - y - 1)) + r' 
  circle [ 
    SVGAttr.Cx x'
    SVGAttr.Cy y'
    SVGAttr.R r
    SVGAttr.Fill colour 
    SVGAttr.Stroke Colours.positionHighlight
  ] [ ]

let viewGridCell x y colour = 
  rect [ 
    SVGAttr.X (cellSize * x) 
    SVGAttr.Y (cellSize * (height - y - 1))
    SVGAttr.Width (cellSize - cellSpacing)
    SVGAttr.Height (cellSize - cellSpacing)
    SVGAttr.Rx cellSpacing
    SVGAttr.Ry cellSpacing
    SVGAttr.Fill colour 
  ] [ ]  

let viewGreedyPolicyCell model x y =
  let directions = getGreedyPolicy model.Task.ActionCount (x,y) model.ActionValueTable
  directions 
  |> List.map (viewArrow x y) 

let viewPath pathHistory =  
  let r' = float (cellSize - cellSpacing)/2.   // position offset of the centre
  pathHistory
  |> List.pairwise
  |> List.collect (fun ((x1, y1), (x2, y2)) -> 
    let x1' = float cellSize * float x1 + r'
    let y1' = float (cellSize * (height - y1 - 1)) + r' 
    let x2' = float cellSize * float x2 + r'
    let y2' = float (cellSize * (height - y2 - 1)) + r' 
    [
      circle [ 
        SVGAttr.Cx x1'
        SVGAttr.Cy y1'
        SVGAttr.R 1.
        SVGAttr.Fill Colours.positionColour 
        SVGAttr.Stroke Colours.positionColour
      ] [ ]
      circle [ 
        SVGAttr.Cx x2'
        SVGAttr.Cy y2'
        SVGAttr.R 1.
        SVGAttr.Fill Colours.positionColour 
        SVGAttr.Stroke Colours.positionColour
      ] [ ]
      line [
        X1 x1'
        Y1 y1'
        X2 x2'
        Y2 y2'
        SVGAttr.Stroke Colours.positionColour
        SVGAttr.StrokeWidth 1
      ] []
    ]
    )

let viewGridWorld (model: Model) =
  svg [
    SVGAttr.Width (width * cellSize)
    SVGAttr.Height (height * cellSize)
  ] [
    for x in 0..width-1 do
      for y in 0..height-1 do
        if (x,y) = model.Task.TerminalState then yield viewGridCell x y Colours.endColour
        else if (x,y) = model.Task.InitialState then yield viewGridCell x y Colours.startColour
        else if model.Task.Cliff |> Array.contains (x,y) then yield viewGridCell x y Colours.cliffColour
        else yield viewGridCell x y Colours.gridColour

    if model.ShowGreedyPolicy then
      for x in 0..width-1 do
        for y in 0..height-1 do
          if ((x,y) <> model.Task.TerminalState) && (model.Task.Cliff |> Array.contains (x,y) |> not) then 
            yield! viewGreedyPolicyCell model x y       

    yield! viewPath model.PathTaken

    let a,b = model.Position 
    yield viewPosition a b Colours.positionColour        
  ]

let view (model:Model) dispatch =
  div [] [
    h1 [] [ str ((model.Algorithm |> string) + " algorithm") ]
    viewGridWorld model
    br []
    str (sprintf "Reward: %.0f" model.TotalReward)
    br []
    str (sprintf "Steps: %d" model.Steps)
    br []
    button [ OnClick (fun _ -> dispatch Step) ] [ str "Step" ]
    br []
    button [ OnClick (fun _ -> dispatch RunEpisode) ] [ str "Run episode" ]
    br []
    button [ OnClick (fun _ -> dispatch ShowPolicy) ] [ str "Show policy" ]
    br []
    button [ OnClick (fun _ -> dispatch NewEpisode) ] [ str "New episode" ]
    br []
    br []
    button [ OnClick (fun _ -> dispatch Reset) ] [ str "Reset" ]
    br []
    button [ 
      OnClick (fun _ -> dispatch (SetAlgorithm Random)); 
      Style [ CSSProp.BackgroundColor (if model.Algorithm = Random then "grey" else "white") ] ] [ str "Random" ]
    button [ OnClick (fun _ -> dispatch (SetAlgorithm QLearning)); 
      Style [ CSSProp.BackgroundColor (if model.Algorithm = QLearning then "grey" else "white") ]  ] [ str "Q-learning" ]
    button [ OnClick (fun _ -> dispatch (SetAlgorithm Sarsa)); 
      Style [ CSSProp.BackgroundColor (if model.Algorithm = Sarsa then "grey" else "white") ]  ] [ str "Sarsa" ]
  ]
  
// App
Program.mkProgram init update view
|> Program.withReact "elmish-app"
|> Program.withConsoleTrace
|> Program.run
