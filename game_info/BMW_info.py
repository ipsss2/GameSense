
BMW_info={
    'Game_info':'',

    'Global_task':'''1. **Explore the Map**:
   - Follow roads and pathways.

2. **Combat with Monsters**:
   - Detect and engage approaching monsters.

3. **Interact with Objects**:
   - Identify objects with white dots above them, approach, and interact.''',

    'UI_info':'',#

    'Frame_attention':'''
1. **Paths**:
   - Describe any currently available paths and their direction or position (e.g., "a small path heading north").

2. **Interactable Object Descriptions**:
   - Check the screen for the following interactable objects and describe their quantity, direction, and position if present:
     - **Herbs** (marked with white-grey icons): e.g., "2 herbs to the northwest."
     - **Guiding Light Beams**: Describe the direction the guiding light is pointing (e.g., "a beam of light pointing north").
     - **Enemies**: If enemies are present:
            Analyze enemies type: Boss (large size, wide arena) or Mobs (regular enemies)
            Than describe their quantity, direction, and position (e.g., "enemy type: Mobs, 3 enemies to the east ahead" or "enemy type: Boss, 1 enemies in circular arena").
     - **Treasure Chests**: If treasure chests are present, describe their quantity and location (e.g., "1 chest to the west").
    - **Shrines marked with white-grey icons**: e.g., "1 shrine with white-grey icons to the southeast."
    ''',#- **Shrines** (marked with white-grey icons): e.g., "1 shrine to the southeast."

    'additional_task_info4_task_plan':'''
        About walk tips:
        - In the game there are invisible walls. If you keep walking but your position doesn't change, try walking in a different direction.
        - When you are stuck in the terrain, you can try walking backwards (S)
        
        About interact with items:
        - Shrine marked with white-grey icons：Interacting with the shrine will bring you to the UI interface, where you can choose to reply to the character's status, but enemies will be refreshed. Note that if you don't know what the UI interface looks like, the task should end with the action of interacting with the shrine.
        - When the interactive object displays the 'E' icon, it can interact directly without walking
        - If repeated interactions with items (such as shrines, flowers, and plants) have not been successful, it is that the item cannot be interacted with and **skip it**
        
        About fight tip:
        - Fighting against enemies is a difficult task, so when you see a mob, plan the action as' FIGHT ' is enough, and when you see a boss, plan the action as 'TRAIN' is enough
        
        About Camera View Adjustment:
        - Camera view directly determines character movement direction (W key movement direction). Camera view affects the agent's ability to perceive and analyze the game environment
        - Camera adjustment **MUST** be treated as an isolated, single-step task. Due to inability to observe real-time screen changes, adjustments must be completed independently before proceeding with game evaluation
        - **STRICTLY** limited to to these two tasks only you can adjust camera view:
        A. Character Direction Control (Horizontal Rotation Only):
            - Task Description: Adjusts character's facing direction to facilitate game exploration. Only horizontal rotation is permitted.
            - Task Design Requirements: Must explicitly specify rotation direction (left/right). Must specify exact rotation angle in degrees
            - Applicable scenarios:
              * Post-combat reorientation when direction has significantly changed
              * At road intersections to capture complete path information
              * When stuck at map edges or terrain
              * When movement direction deviates from intended path
            
        B. View Perspective Adjustment:
            -Task Description: Adjusts camera angle when view is suboptimal (too high/low/askew). Ensures clear visibility of road and distant features.
            - Task Design Requirements: Must explicitly specify adjustment direction (up/down/left/right).
            - Correct when:
              * View is too high (excessive sky/trees visible)
              * View is too low (excessive ground visible)
              * Horizontal view is incomplete (partial road visibility)
              * Road's distant features are unclear
            - Applicable scenarios:
              * Post-combat view restoration
              * After character direction changes
              * When stuck in terrain to identify escape routes
              * When distant path visibility is poor
            
        Technical Parameters:
        - Horizontal rotation: ~36° per adjustment
        - Vertical rotation: ~25° per adjustment

        ''',
    'control_info':'''Game Control Mapping Overview:
This is a third-person action RPG control scheme. All controls support continuous press for sustained actions, or quick taps for instant actions.

Basic Movement:
W: Move forward
S: Move backward
A: Move left
D: Move right

Special Actions:
SPACE: Dodge/Roll
E: Interact with objects
R: Drink potion to restore health

Function Keys:
STOP: Stop game
ESC: Quit Menu

Special Training Process：
TRAIN: Start the RL training process and build the ability to defeat boss
FIGHT: Use existing combat programs to defeat mobs

UI Navigation:
UP: UI control: up    
DOWN: UI control: down
LEFT: UI control: left
RIGHT: UI control: right
ENTER: Select in UI interfaces

Turn Camera in Game:
VIEW_LEFT: Camera Left 36 degree,
VIEW_RIGHT: Camera Right 36 degree,
VIEW_UP: Camera Up 20 degree,
VIEW_DOWN: Camera Down 20 degree

Timing Guidelines:
- Basic movements: 
Approximately 0.8~1 second per protagonist height distance
    - Examples:
      * Moving 0.5x character height ≈ 0.5 seconds
      * Moving 3.5x character height ≈ 3~3.5 seconds 
      * Moving 6x character height ≈ 5~6 seconds 
      
Camera View Adjustment Guidelines:
- Camera view directly determines character movement direction (W key movement direction). Camera view affects the agent's ability to perceive and analyze the game environment
- Camera adjustment **MUST** be treated as an isolated, single-step task. Due to inability to observe real-time screen changes, adjustments must be completed independently before proceeding with game evaluation
- **STRICTLY** limited to to these two tasks only you can adjust camera view:
A. Character Direction Control (Horizontal Rotation Only):
    - Task Description: Adjusts character's facing direction to facilitate game exploration. Only horizontal rotation is permitted.
    - Task Design Requirements: Must explicitly specify rotation direction (left/right). Must specify exact rotation angle in degrees
    - Applicable scenarios:
      * Post-combat reorientation when direction has significantly changed
      * At road intersections to capture complete path information
      * When stuck at map edges or terrain
      * When movement direction deviates from intended path
    
B. View Perspective Adjustment:
    -Task Description: Adjusts camera angle when view is suboptimal (too high/low/askew). Ensures clear visibility of road and distant features.
    - Task Design Requirements: Must explicitly specify adjustment direction (up/down/left/right). Without description of degrees.
    - Correct when:
      * View is too high (excessive sky/trees visible)
      * View is too low (excessive ground visible)
      * Horizontal view is incomplete (partial road visibility)
      * Road's distant features are unclear
    - Applicable scenarios:
      * Post-combat view restoration
      * After character direction changes
      * When stuck in terrain to identify escape routes
      * When distant path visibility is poor
    
Technical Parameters:
- Horizontal rotation: ~36° per adjustment
- Vertical rotation: ~25° per adjustment

3. Usage Restrictions:
- Avoid adjustments during normal movement

Remember: Camera view adjustment should be used sparingly and only when necessary. Each adjustment should be purposeful and wait for feedback before proceeding with further actions."

  
Note: Adjust timing based on terrain and obstacles. The actual duration may need slight modifications for diagonal movement or complex paths.
You should also refer to Similar Action Experiences for specific numerical settings.
        
- Skill/Attacks: Quick press 0.1 per 
- When the interactive object displays the 'E' icon, it can interact directly without walking

''',#

# Core Combat:
# MOUSE1: Light Attack
# MOUSE2: Heavy Attack
# MOUSE3: Target Enemy Lock
# Spell Skills:
# 1: Fix enemies' body

# - The trade-off between non real time Agent and safety: When designing tasks, consider the potential security issues that may arise from your inability to interact with the game in real time. Flexibly apply the STOP state at the end of task, such as choosing to enter the stop state after fighting against mobs
# - If there are enemies in the environment, it is recommended to end action_code with STOP for safety.
    'additional_action_info':'''
If you are performing a task to attack mobs or bosses, these suggestions will help you output a reasonable action_code:

1. For Regular Mobs:
   - Approach the Mobs location
   - combat programs to fight ('FIGHT', 0)
   - ('FIGHT', 0) is a complete combat program, so there is no need to follow anything else afterwards
   - Example sequence: [('W', duration_to_reach_boss), ('FIGHT', 0)]
   
2. For Boss:
   - Approach the boss location
   - Start RL training mode using ('TRAIN', 0)
   - ('TRAIN', 0) is a complete RL training program, so there is no need to follow anything else afterwards 
   - Example sequence: [('W', duration_to_reach_boss), ('TRAIN', 0)]

3. For View Perspective Adjustment：
   - if task is： Camera view need to turn left 150 degrees
   - because Horizontal rotation has 36° per adjustment, we need adjust Camera view to left 5 times
   - Example sequence: [('VIEW_LEFT', 0.1), ('VIEW_LEFT', 0.1), ('VIEW_LEFT', 0.1), ('VIEW_LEFT', 0.1), ('VIEW_LEFT', 0.1)]

4. Character Direction Control：
   - if task is： Camera view need to be adjusted upwards and to the left
   - We only need adjust Camera view in the corresponding direction once
   - Example sequence: [('VIEW_UP', 0.1), ('VIEW_LEFT', 0.1)]
    ''',

    'Mapping_info': {
    'W': 'Move forward',
    'S': 'Move backward',
    'A': 'Move left',
    'D': 'Move right',
    'SPACE': 'Dodge/Roll',
    'E': 'Interact with objects',
    'R': 'Drink potion to restore health',
    'ESC': 'Stop game and enter Menu/Quit Menu',
    'ENTER': 'Confirm selection in UI',
    'TRAIN': 'Start RL training process',
    'FIGHT': 'Use existing combat programs to defeat mobs',
    'UP': "UI control: up",
    'DOWN': "UI control: down",
    'LEFT': "UI control: left",
    'RIGHT': "UI control: right",
    'STOP': "Stop game",
    'VIEW_LEFT': 'Camera Left 36 degrees',
    'VIEW_RIGHT': 'Camera Right 36 degrees',
    'VIEW_UP': 'Camera Up 20 degrees',
    'VIEW_DOWN': 'Camera Down 20 degrees'
}
}
