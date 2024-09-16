+++
title = 'Program'
draft = "false"
layout='program'
+++

## Welcome to Agimus Winter School !

Agimus is a European research program founded by Horizon Europe, coordinated by LAAS-CNRS (FR) and gathering Inria (FR), CTU (CZ), PAL-Robotics (SP) and other industrial partners, studying the next generation of AI-powered robots for agile production. The consortium developed an internationally recognized expertise in motion planning, optimal control, computer vision, reinforcement learning, force control and robot design, along with mature software to implement them. The objective of the school is to participate in the dissemination of our expertise, by training students involved in the consortium and external students in a 5-day event in the South of France.

## Schedule

The (final) schedule of the week is below.

![Scenario 1: Across columns](/schedule_v2.png)

 

## Main teachers

The teachers are mostly key researchers and engineers of the Agimus consortium:

- Nicolas Mansard (LAAS-CNRS), project coordinator, expert in numerical optimal control and robot modeling
- Florent Lamiraux (LAAS-CNRS), expert in motion planning 
- Josef Sivic, Vladimir Petrik, and Mederic Fourmy (CTU), experts in computer vision
- Justin Carpentier (Inria), expert in numerical programming, optimal control, robot modeling, and guru of the robot modeling software Pinocchio
- Luca Marchionni and Narcis Miguel i Baños (PAL), experts in robot design and control
- Guilhem Saurel (LAAS-CNRS), expert in software development and deployment for robotics

Agimus winter school also involves the participation of PhD students:

- Wilson Jallet (LAAS-CNRS & Inria), a 3rd year PhD student in numerical optimal control and optimization
- Louis Montaut (CTU & Inria) and Quentin Le Lidec (Inria), 3rd year PhD students working on simulation of robotics systems through contacts

The school will also feature keynote speeches by invited researchers, among which:

- Ludovic Righetti (NYU), expert in robot learning, optimal control and force control
- Adrien Taylor (Inria), expert in optimization and numerical programming 
- Timothy Bretl (UIUC), expert in motion planning

## Main topics

The topics will cover the main advanced methods to program and control a versatile mobile robot in a manufacturing environment.
#### Simulation: dynamics and collisions

This class will cover everything from the classical rigid dynamic algorithms (as popularized by Roy Featherstone and efficiently implemented in the software Pinocchio), contact modeling and efficient resolution, computation of collision and distances between geometric primitives, to the complete simulation of a rigid polyarticulated system in unilateral contact with its environment. The class will be supported by practical work sessions on the laptops of the participants using Pinocchio and HPP-FCL.

- Session 1: rigid body dynamic models and algorithms
- Session 2: solving the contact dynamics
- Session 3: collision algorithms

With a keynote speech of Ludovic Righetti.
#### Optimal control: formalization and solvers

This class will cover the field of numerical optimal control for robotics, i.e how to understand an optimal control problem, transcribe it into a proper static nonlinear optimization problem subject to hard constraints and sparse structure, and solve it with cutting edge augmented Lagrangian algorithms. The class will exploit the new Aligator software developed as a follow-up of the widespread Crocoddyl library.

- Session 1: basics of numerical optimal control, from the formulation of the problem to efficient resolution using differential dynamic programming
- Session 2: basics of constraint programming and explanation of optimal-control solver exploiting augmented-Lagrangian strategies

With a keynote speech of Adrien Taylor.
#### Task and motion planning

Robot planning becomes seriously hard when the IA has to simultaneously decide a sequence of tasks to be achieved (discrete decision) and the continuous movement to achieve them. The class will cover the topic from the basic motion planning algorithms (RRT, etc) to the recent progress to solve the task-and-motion planning problem. The class will be supported by the motion-planning framework HPP, already successfully used on a wide range of applications, from humanoid robotics scenarios to industrial problems at Airbus.

- Session 1: basics of motion planning
- Session 2: advanced algorithms for task and motion planning

With a keynote speech by Tim Bretl.
#### Computer vision: object recognition and tracking

A key difficulty of computer manipulation for robotics vision is the demanded rate of success, accuracy and speed to implement successful vision-based controllers. This class will propose to approach the problem using render-and-compare approaches as popularized in robotics by the object detector CosyPose. It will be supported by the software HappyPose, which reimplemented the award-winning algorithm CosyPose and its successor MegaPose in a clean and easily available package. It will cover a quick survey of the landscape of computer vision for robot manipulation, explain the principle of render-and-compare, detail the CosyPose and MegaPose algorithms and propose a practical session to test them. It will also open to new research directions to move from these detectors to proper object trackers needed to control dynamic robot movements.

With a keynote speech by Josef Sivic.
#### ROS2 control for the new Tiago Pro

Agimus participated in the design and release of the new robot from PAL robotics, Tiago Pro, unveiled at ICRA 2023. This robot will be delivered with a new ROS2-control driver, with a new concept of low-level control which will allow the researchers to access the very motor control servo and unleash the performances of their high-level controllers in particular in contact and for haptic-and-vision feedback.

The course will be a practical hands-on session to learn how to control Tiago Pro in simulation. Ideally, we will open the class to a hackathon on the real robot platform. The exact organization of the access to the real robot hardware remains to be defined, to account for the constraints imposed by the hosting institution.
#### Principles of efficient software development for robotics

The Gepetto team of LAAS-CNRS, now joined by the Willow team of Inria under the leadership of Justin Carpentier, are recognized worldwide for the quality (and quantity) of open-source software produced and used by our collaborators. We believe that ethical science can only be enabled by the highest standard of openness, by allowing our peers to reproduce our results and capitalize on our explorations. This process is not cheap, and requires significant effort of our teams and high engineering standards. Yet a lot remains to be done in the quality of academic software, including our own work. In this class, we will provide the main principles to ease the publication of your own work and enable easy reproduction, deployment and collaboration.

## Important dates

Registrations are open until the 12th of November. Payment will be set up during the following week (deadline 17th of November).

We are expecting ~30 students from within the Agimus project and ~20 students from outside. The selection will be mainly on a first-come-first-served basis, with some few adjustments by the moderator team to avoid overrepresentation. We will publish a first status of participation on 1st Nov.


The school then start on Monday 11/dec, with students mostly expected to arrive at previous Sunday night.

## Location and venue

The school will be hosted by the Oceanological Observatory of Banyuls-sur-Mer (https://www.obs-banyuls.fr/en/oob.html), near Perpignan, in between  Toulouse (FR) and Barcelona (SP). 

 
![Scenario 1: Across columns](/banyuls.png)
*Banyuls, 50km north of the French-Spanish border, on the Mediteranean cost*

 

![Scenario 1: Across columns](/banyuls_image2.png)
*Hotel ("centre d'hebergement"), observatory and classrooms ("Université P&M Curie")*

 

The training center is composed of an amphitheater, several classrooms, a hotel, a restaurant, all that on the harbor of the small town. 

You can best join Banyuls by train, with high-speed trains from Paris and Barcelona to Perpignan and then a quick local ride to Banyuls. You can also reach the site from Barcelona airport (although please consider a 0%-flight trip as a first choice: we will offer the gala dinner to the participants who will not use the plane to reach us). 

The price of the school includes accommodation,  food. Teachers are freely serving as part of their participation to the Agimus project. 

 

## Princing

The prices are only valid for student outside the Agimus project.

Basic price: 800EUR (with tax "TTC")

Extra night: 100EUR (with tax "TTC")

The basic cost covers: 5 nights from Sunday 10/dec to Friday 15/dec (extra night possible to extend to Saturday 16/dec), food from Monday morning to Friday afternoon (extra dinner on Friday evening with the extra-night ticket), social events, classes and a part of the expenses of the teachers and the keynote speakers.

For participant inside the consortium Agimus, you will be asked to only pay the hotel and food at the conference (two separated bills); the project covers the rest of the expenses, corresponding to a total of approximately 700EUR (tax free).

For Agimus PIs coming only to the Tuesday-Wednesday meeting, your registration corresponds to one night and food for the two days. Extra night with the corresponding food can be booked if you want to arrive on Monday evening or leave Thursday morning.

 

## Supported by ...


![Scenario 1: Across columns](/winter-school-2023/eurobin_small.png)

![Scenario 1: Across columns](/winter-school-2023/agimus_small.png)

![Scenario 1: Across columns](/winter-school-2023/laas_dpt.png)

![Scenario 1: Across columns](/winter-school-2023/eu_320.png)
