import numpy as np
import copy
import time

class Vec3d(np.ndarray):
    """
    Convenient and specific wrapper for a 3d np vector with addition / subtraction / division operations inherited
    """
    def __new__(cls,lst=(0,0,0)):
        """
        Inherits from the np.ndarray class, which was initialised in __new__ instead of __init__.
        """
        ar = np.array(lst,dtype=float)
        x = np.ndarray.__new__(cls,shape=(3,),dtype=float,buffer=ar) 
        # x is now a Vec3d object instead of a ndarray object
        return x
    def length(self):
        """Returns the norm of the vector."""
        return np.linalg.norm(self)
    def __mul__(self,other):
        """Defines the * operator as the dot product."""
        return np.dot(self,other)
    def unitvec(self):
        """Returns the unit vector."""
        return self/self.length()

class LJ:
    """The Lennard-Jones potential."""
    def __init__(self,re,de):
        self.re = float(re)
        self.de = float(de)

    def __call__(self,r):
        """
        LJ14 = LJ(1,4) will define a new Lennard-Jones potential function,
        LJ14(r) returns the value of the function at r.
        """
        k = self.re/r
        return 4*self.de*(k**12-k**6)

class Morse:
    """The Morse potential."""
    def __init__(self,re,de):
        self.de = float(de)
        self.re = float(re) # In sigma units

    def __call__(self,r):
        return self.de*((1-np.exp(-(r-self.re)))**2-1) # The -1 at the end is to set all attractive energy to negative values, so that it coheres with the definition of Lennard-Jones potential

class System:
    """
    This initialises a system of n particles governed by a potential, passed in as a callable Python function
    """
    def __init__(self,n,potential):
        self.positions = {}
        self.box_size = 3 # A good compromise between likelihood of explosion (small box size) and convergence speed (large box size)
        self.positions[0] = Vec3d(np.random.uniform(-self.box_size,self.box_size,(3,)))
        """
        The following for loop initiailises the particles in a confirguration that 
        doesn't result in explosion, which is defined as having a non-negative pairwise potential energy
        """
        self.n = n
        self.pot = potential
        for i in range(1,self.n):
            num_non_explode = 0
            while num_non_explode != i:
                self.positions[i] = Vec3d(np.random.uniform(-self.box_size,self.box_size,(3,)))
                for j in range(i):
                    diff = self.positions[j] - self.positions[i]
                    dist = diff.length()
                    if self.pot(dist) < 0:
                        num_non_explode += 1
                    else:
                        pass
        self.pe = 0 # The potential energy of the system
        self.step = 0
        self.lamb = {}
        self.lamb_ini = 5e-4
        self.lamb_max = self.lamb_ini*5e4
        self.lamb_min = self.lamb_ini/100
        for atom in self.positions.keys():
            self.lamb[atom] = self.lamb_ini # This is the initial learning rate

    def pot_eval(self,pos):
        """
        This evaluates the potential energy of the system at any given configuration. Instead of using the self.positions dictionary the position dictionary needs to be passed as an argument (even if it's self.positions) because it is used to evaluate finite differences in which one atom moves by a small step from self.positions.
        """
        scalar_dist = {}
        for i in range(1,self.n):
            for j in range(i):
                vec_dist = pos[i]-pos[j]
                scalar_dist[(i,j)] = vec_dist.length()
        potential_sum = 0
        for key in scalar_dist.keys():
            potential_sum += self.pot(scalar_dist[key])
        return potential_sum
    
    def grad_eval(self,grad_old):
        """
        There are two problems with a fixed step size 
            1. Atomic potentials all have this 'explosive' behaviour near r=0, where two atoms in close proximity will explode apart (very large gradient => end up too far away to drag back). 
            2. For atoms that are far apart from each other it takes very long to build up acceleration. (This is especially true for LJ as it vanishes rapidly past eqm length. This is expected as it's mostly used to model noble gases)
        The first can be most effectively addressed by imposing a movement cap - here set at 0.1, to 'slow down' explosions. This is implemented in System.gd_time_stepper().
        The second calls for adaptive step sizes, which was most easily done with the Rprop algorithm, where if the updates are always in the same direction, it is multiplicatively increased up to a maximum, and if it backtracks it is braked down to a minimum value. In this way it helps with the first problem as well.

        Overall the update vector is 
        v_{i,t} = gamma*v_{i,t-1} + lamb_{i}*gradient_{i,t}
        """
        grad = {}          
        h = float(1e-8) # The small constant for computing finite differences
        gamma = 0.6 # This is the dampening/momentum constant, usual range 0.8<gamma<0.99 in machine learning, however here it was found 0.6 worked well.
        inc = 1.1 # Multiplicative increase constant for lambda.
        dec = 0.05 # This may look excessively small but larger values can cause numerical instability.
        if self.step == 0:
            grad_old = {atom : Vec3d((0,0,0)) for atom in self.positions.keys()}
            # No old grad dictionary in the first step, just initialize to all zeroes.
        for atom in self.positions.keys():
            # This uses the second-order central difference method, with O(h^2) 
            grad_vec = Vec3d((0,0,0))
            for uvec in [0,1,2]:
                upperdiff = copy.deepcopy(self.positions) # Deepcopy must be used since the dict is keyed by a tuple, i.e., it's nested
                upperdiff[atom][uvec] += h
                lowerdiff = copy.deepcopy(self.positions)
                lowerdiff[atom][uvec] -= h
                grad_vec[uvec] = (self.pot_eval(upperdiff) - self.pot_eval(lowerdiff))/(2.0*h)
            """
            This is an implementation of Rprop (Resilient backpropagation), in which if the new gradient update of an atom has a positive dot product, 
            i.e., roughly in the same direction, with the previous gradient update, then the learning rate increases, otherwise it decreases. 
            The 'braking' should be a lot faster than the 'acceleration' as braking happens near the bottom of the well, which could lead to sudden explosion of the system.
            """
            if grad_vec * grad_old[atom] > 0:
                self.lamb[atom] = min(self.lamb[atom] * inc, self.lamb_max)
            elif grad_vec * grad_old[atom] < 0:
                self.lamb[atom] = max(self.lamb[atom] * dec, self.lamb_min)
            else:
                pass
            grad[atom] = self.lamb[atom]*grad_vec + gamma*grad_old[atom]
        return grad
    
    def gd_time_stepper(self):
        """
        Interfaces with the LandscapeExplorer class, takes no argument and takes the system one time step ahead. 
        This runs until convergence, which is when the average absolute change in potential energy of the system is less than a small number over 100 steps.
        The time stepper also exits when the system is stuck at a high-energy configuration, which happened during debugging but now it rarely if ever happens, left in just in case.
        """
        converge = False
        grad_old = {}
        movement_cap = 0.1 # Prevents explosion
        self.pe = self.pot_eval(self.positions) # Set self.pe to the correct initial value
        delta_pe = 0
        sum_pe = 0
        while not converge:
            pe_old = self.pe
            print(pe_old)
            grad = self.grad_eval(grad_old)
            self.step += 1
            print(f'Step {self.step}')
            for atom in self.positions.keys():
                if grad[atom].length() > movement_cap:
                    grad[atom] = movement_cap * grad[atom].unitvec()
                self.positions[atom] -= grad[atom]
            grad_old = copy.deepcopy(grad)
            self.pe = self.pot_eval(self.positions)
            delta_pe += abs(pe_old - self.pe)
            sum_pe += pe_old
            if self.step % 100 == 0:
                if delta_pe/100 < 1e-5:
                    converge = True
                if sum_pe/100 > 100:
                    print('The system is stuck in a high-energy configuration.')
                    converge = True
                sum_pe = 0
                delta_pe = 0
    
    def output_xyz(self,prefix):
        """
        Outputs the system to a .xyz file, accepts a string as argument to prefix the filename. This is usually the energy of the system. The filename looks like
        -73.69_7_LJ_re1.0.xyz
        meaning this depicts a system of 7 particles governed by a LJ potential with sigma=1.0, which reached equilibrium at -73.69 epsilons.
        """
        pot_re = self.pot.re
        with open(f'{prefix}_{self.n}_{type(self.pot).__name__}_re{self.pot.re}.xyz','w') as f:
            f.write(f'{self.n}\n')
            f.write(f'Optimum geometry of {self.n} atoms governed by {type(self.pot).__name__} potential\n')
            for atom in self.positions.keys():
                f.write(f'Atom_{atom}\t')
                for i in self.positions[atom]:
                    f.write(f'{i}\t')
                f.write('\n')

class LandscapeExplorer:
    """
    Initialised by n, the number of particles, and pot, the potential as a python function.

    This is the user interface, where the user can specify the number of particles and the potential that govern them. This is advantageous because in case many stable local minima exist (i.e., frequent kinetic trapping) the system can become stuck and it's good to reinitialise the system over and over again to explore all possible stable conformations. This is easily implemented by adding a while True: clause. But because the current algorithm doesn't need many restarts to reach the global minimum this isn't implemented.
    """
    def __init__(self,n,pot):
        self.n = n
        self.pot = pot

    def scheduler(self):
        sys = System(self.n,self.pot)
        t0 = time.time()
        sys.gd_time_stepper()
        t1 = time.time()
        t_avg = (t1-t0)/sys.step
        print(t_avg)
        conf_pe = round(sys.pe,2)
        sys.output_xyz(f'{conf_pe}')

print('=======================================')
print(u'| Welcome to Brian\'s Energy Minimizer! |')
print('|        Christ\'s College, zz376       |')
print('|           Part II Chemistry          |')
print('|    Programming Practical Exercise 4  |')
print('=======================================')

while True:
    prompt = 'Please choose from the following options:\n   1. 7 particles governed by a Lennard-Jones potential with sigma = 1 and epsilon = 4\n   2. 7 particles governed by a Morse potential with re/sigma = 1 and De = 4\n   3. 7 particles governed by a Morse potential with re/sigma = 2 and De = 4\n   4. Custom\n'
    choice = int(input(prompt))
    if choice == 1:
        LandscapeExplorer(7,LJ(1,4)).scheduler()
    if choice == 2:
        LandscapeExplorer(7,Morse(1,4)).scheduler()
    if choice == 3:
        LandscapeExplorer(7,Morse(2,4)).scheduler()
    if choice == 4:
        num_particles = int(input('Number of particles:\n'))
        potential = int(input('1. Lennard-Jones\n   2. Morse\n'))
        scale_length = float(input('Scale length (e.g sigma for LJ and re/sigma for Morse)\n'))
        scale_energy = float(input('Scale energy (e.g. epsilon for LJ and De for Morse)\n'))
        if potential == 1:
            LandscapeExplorer(num_particles,LJ(scale_length,scale_energy)).scheduler()
        if potential == 2:
            LandscapeExplorer(num_particles,Morse(scale_length,scale_energy)).scheduler()