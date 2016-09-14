FROM andrewosh/binder-base

MAINTAINER Ryan Dwyer <ryanpdwyer@gmail.com>

USER root

# Add Julia dependencies
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y git make gcc gfortran wget
RUN apt-get install -y software-properties-common
RUN apt-get install -y julia

USER main

# Install Julia kernel
RUN julia -e 'Pkg.add("IJulia")'
RUN julia -e 'Pkg.add("ODE")'
RUN julia -e 'Pkg.add("PyPlot")'
