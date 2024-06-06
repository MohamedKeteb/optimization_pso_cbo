import numpy as np
import numpy.random as rnd



def update_velocity(particles, velocity, pbest, gbest):
    
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    inertia = w * velocity
    cognitive = c1 * r1 * (pbest - particles)
    social = c2 * r2 * (gbest - particles)
    new_velocity = inertia + cognitive + social

    idx = np.where(new_velocity > vmax)
    new_velocity[idx] = vmax
    idx = np.where(new_velocity < -vmax)
    new_velocity[idx] = -vmax

    return new_velocity

#------------------------------------------------------------


def update_position(particles, velocity):
    
    new_particles = particles + velocity
    idx = np.where(new_particles > x_sup)
    new_particles[idx] = x_sup
    idx = np.where(new_particles < x_inf)
    new_particles[idx] = x_inf

    return new_particles

#------------------------------------------------------------



def update_best(pbest, gbest, particles):
    
    obj = function(particles)
    if obj.min() < function(gbest):
        gbest = particles[np.argmin(obj)]

    idx = np.where(obj < function(pbest))
    pbest[idx] = particles[idx]
    return pbest, gbest


#------------------------------------------------------------



def run_pso(T):

    particles = np.random.uniform(x_inf, x_sup, N)
    velocity = np.zeros_like(particles)
    pbest = np.copy(particles)
    gbest = pbest[np.argmin(function(pbest))]
    dynamic = [np.copy(particles)]

    for _ in range(T):
    
        velocity = update_velocity(particles, velocity, pbest, gbest)
        particles = update_position(particles, velocity) 
        dynamic.append(np.copy(particles))
        pbest, gbest = update_best(pbest, gbest, particles)

    return gbest


#--------------------------------------------------------------





    
    
    