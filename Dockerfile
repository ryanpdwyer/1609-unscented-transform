FROM andrewosh/binder-base

MAINTAINER Ryan Dwyer <ryanpdwyer@gmail.com>

USER root

# Add Julia dependencies
RUN apt-get update
RUN apt-get -y install build-essentials
RUN apt-get -y git make gcc gfortran wget

USER main

RUN wget https://github.com/JuliaLang/julia/releases/download/v0.4.6/julia-0.4.6-full.tar.gz
RUN tar -zxvf julia-0.4.6-full.tar.gz
RUN cd julia-0.4.6-full
RUN make
RUN pwd
RUN ln -s /users/main/julia-0.4.6-full/julia /usr/local/bin/julia
RUN hash -r


# Install Julia kernel
RUN julia -e 'Pkg.add("IJulia")'
RUN julia -e 'Pkg.add("ODE")'
RUN julia -e 'Pkg.add("PyPlot")'
