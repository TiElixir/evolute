"""
evolution/race.py

Visual race mode for multiple creatures.
Generates a population, runs them simultaneously in one space without colliding,
selects the best (furthest X), mutates to form next generation.
"""

import os
import sys
import time
import math
import random
import pygame
import pymunk
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creature.morphology import Genome
from creature.builder import build_creature, Creature, PIXELS_PER_METER
from evolution.population import Population
from evolution.genome_ops import (
    mutate_bone_size, mutate_joint_params, add_random_bone,
    remove_random_leaf_bone, toggle_motor
)
from rl.networks import ActorCritic

GROUND_CATEGORY = 1 << 31

class ParallelCreatureEnv:
    def __init__(self, genomes, config):
        self.genomes = genomes
        self.config = config
        self.physics_cfg = config.get("physics", {})
        self.ground_y = float(self.physics_cfg.get("ground_y", 50))
        self.space = pymunk.Space()
        self.space.gravity = (0, self.physics_cfg.get("gravity", -900))
        self.space.damping = 0.9

        # Ground
        self.ground_segment = pymunk.Segment(self.space.static_body, (-50000, self.ground_y), (50000, self.ground_y), 5)
        self.ground_segment.friction = 1.0
        self.ground_segment.elasticity = 0.0
        self.ground_segment.filter = pymunk.ShapeFilter(categories=GROUND_CATEGORY)
        self.space.add(self.ground_segment)

        self.creatures = []
        spawn_h = self.config.get("spawn_height_above_ground", 250)
        spawn_y = self.ground_y + spawn_h
        
        # Spawn creatures with non-colliding filters
        for i, genome in enumerate(genomes):
            # Stagger spawn slightly so they don't exactly overlap visually
            spawn_x = self.config.get("spawn_x", 300) + (i % 3) * 50
            creature = build_creature(self.space, genome, position=(spawn_x, spawn_y))
            creature.space = self.space
            
            cat = 1 << i
            mask = cat | GROUND_CATEGORY
            for shape in creature.shapes.values():
                shape.filter = pymunk.ShapeFilter(categories=cat, mask=mask)
            
            self.creatures.append(creature)

    def step(self, actions):
        # Apply actions
        for creature, action in zip(self.creatures, actions):
            if action is not None:
                creature.apply_action(np.tanh(action))
                
        # Step physics
        dt = self.physics_cfg.get("dt", 1.0/60.0)
        substeps = self.physics_cfg.get("substeps", 4)
        sub_dt = dt / substeps
        for _ in range(substeps):
            self.space.step(sub_dt)

    def get_observations(self):
        return [c.get_observation() for c in self.creatures]


def mutate_genome(genome: Genome) -> Genome:
    """Apply a random mutation to create a child genome."""
    import copy
    child = copy.deepcopy(genome)
    child.name = f"{genome.name.split('_')[0]}_m{random.randint(1000, 9999)}"
    
    op = random.choice([
        "size", "joint", "add", "remove", "motor"
    ])
    
    for _ in range(10): # retry limit
        try:
            if op == "size": mutate_bone_size(child)
            elif op == "joint": mutate_joint_params(child)
            elif op == "add": add_random_bone(child)
            elif op == "remove": remove_random_leaf_bone(child)
            elif op == "motor": toggle_motor(child)
            return child
        except ValueError:
            # mutation failed validation, retry
            pass
    return child


def run_race(config, args):
    from environment.renderer import Renderer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_creatures = 10
    n_generations = 50
    steps_per_race = 300
    
    seed_path = args.genome
    if not os.path.isfile(seed_path):
        print(f"Seed genome not found: {seed_path}")
        return
        
    seed_genome = Genome.load(seed_path)
    
    # Initialize population by mutating the seed
    genomes = [seed_genome]
    for _ in range(n_creatures - 1):
        genomes.append(mutate_genome(seed_genome))
        
    # We will also keep random neural networks for them
    # To pass down behavior, we would need to pass down weights. 
    # For a purely structural race, random weights each time is okay, 
    # but let's persist networks to make them learn a bit!
    networks = []
    for g in genomes:
        # Dummy build to get dims
        c = Creature(g)
        space = pymunk.Space()
        c = build_creature(space, g)
        ac = ActorCritic(c.observation_dim, c.action_dim, hidden=64, device=device)
        networks.append(ac)
        
    renderer = Renderer(width=1200, height=700, title="Evolutionary Race")
    renderer.init()
    
    for gen in range(n_generations):
        print(f"--- Generation {gen+1} ---")
        env = ParallelCreatureEnv(genomes, config)
        
        for step in range(steps_per_race):
            obs_list = env.get_observations()
            actions = []
            for i, obs in enumerate(obs_list):
                if env.creatures[i].is_fallen(env.ground_y):
                    actions.append(np.zeros(env.creatures[i].action_dim))
                    continue
                    
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    # Evaluate deterministically
                    features = networks[i].trunk(obs_t)
                    mean = networks[i].actor_head(features)
                actions.append(mean.squeeze(0).cpu().numpy())
                
            env.step(actions)
            
            # Find the leader for camera tracking
            distances = [c.get_torso_position()[0] for c in env.creatures]
            leader_idx = np.argmax(distances)
            leader = env.creatures[leader_idx]
            
            if renderer.poll_quit():
                print("Race cancelled by user.")
                renderer.close()
                return

            # Render
            renderer.screen.fill((18, 18, 28))
            renderer._draw_grid()
            renderer._draw_ground()
            for i, c in enumerate(env.creatures):
                renderer._draw_creature(c, None)
            
            # HUD
            renderer._draw_hud({
                "Generation": gen + 1,
                "Step": f"{step}/{steps_per_race}",
                "Leader": genomes[leader_idx].name,
                "Distance": f"{distances[leader_idx] / PIXELS_PER_METER:.2f}m"
            })
            
            # Camera
            torso_x, _ = leader.get_torso_position()
            target_cam = torso_x - renderer.width * 0.35
            renderer.camera_x += (target_cam - renderer.camera_x) * 0.08
            
            renderer.tick(fps=60)
            
        # Race over, evaluate fitness
        distances = [c.get_torso_position()[0] for c in env.creatures]
        best_idx = int(np.argmax(distances))
        best_genome = genomes[best_idx]
        best_net = networks[best_idx]
        print(f"Winner: {best_genome.name} (Distance: {distances[best_idx] / PIXELS_PER_METER:.2f}m)")
        
        # Reproduce
        new_genomes = [best_genome]
        new_networks = [best_net]
        
        for _ in range(n_creatures - 1):
            child_g = mutate_genome(best_genome)
            new_genomes.append(child_g)
            
            # Copy network structure
            c = Creature(child_g)
            space = pymunk.Space()
            c = build_creature(space, child_g)
            child_net = ActorCritic(c.observation_dim, c.action_dim, hidden=64, device=device)
            # We don't copy weights because observation_dim/action_dim might have changed due to mutation
            new_networks.append(child_net)
            
        genomes = new_genomes
        networks = new_networks

    renderer.close()
