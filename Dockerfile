FROM andrewosh/binder-base

MAINTAINER Ryan Dwyer <ryanpdwyer@gmail.com>

USER root

# Add Julia dependencies
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:staticfloat/juliareleases
RUN add-apt-repository ppa:staticfloat/julia-deps
RUN apt-get update
RUN apt-get install -y julia libnettle4 && apt-get clean

USER main

# Install Julia kernel
RUN julia -e 'Pkg.add("IJulia")'
RUN julia -e 'Pkg.add("ODE")'
RUN julia -e 'Pkg.add("PyPlot")'
