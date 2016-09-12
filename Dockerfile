FROM andrewosh/binder-base

MAINTAINER Ryan Dwyer <ryanpdwyer@gmail.com>

USER root

# Add Julia dependencies
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y git make gcc gfortran wget

USER main

RUN git clone git://github.com/JuliaLang/julia.git && \
    cd julia && \
    git checkout release-0.4 && \
    make && \
    pwd  && \
    ln -s /users/main/julia/julia /usr/local/bin/julia && \
    hash -r


# Install Julia kernel
RUN julia -e 'Pkg.add("IJulia")'
RUN julia -e 'Pkg.add("ODE")'
RUN julia -e 'Pkg.add("PyPlot")'
