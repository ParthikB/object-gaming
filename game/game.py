import pygame

from pygame.locals import *

WIDTH, HEIGHT = 800, 600
KEYS          = [False, False, False, False]
PLAYER_POS    = [300, 200]
SPEED         = 2


# Initializing the game
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Load the Images
player = pygame.image.load('./data/player.png')

# Keep looping through
while True:

    # Clearing the screen before drawing again, basically refreshing
    screen.fill(0)

    # Drawing the Screen elements
    screen.blit(player, PLAYER_POS)

    # Update the screen
    pygame.display.flip()

    # Loop through the events
    for event in pygame.event.get():

        # Check if the event is the X button
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)
        
        if event.type == pygame.KEYDOWN:
            if event.key==K_w:
                KEYS[0]=True
            elif event.key==K_a:
                KEYS[1]=True
            elif event.key==K_s:
                KEYS[2]=True
            elif event.key==K_d:
                KEYS[3]=True
        
        if event.type == pygame.KEYUP:
            if event.key==pygame.K_w:
                KEYS[0]=False
            elif event.key==pygame.K_a:
                KEYS[1]=False
            elif event.key==pygame.K_s:
                KEYS[2]=False
            elif event.key==pygame.K_d:
                KEYS[3]=False


    if KEYS[0]:
        PLAYER_POS[1] -= SPEED
    elif KEYS[2]:
        PLAYER_POS[1] += SPEED
    if KEYS[1]:
        PLAYER_POS[0] -= SPEED
    elif KEYS[3]:
        PLAYER_POS[0] += SPEED