FROM lukemathwalker/cargo-chef:0.1.62-rust-1.73-slim-bullseye AS chef
WORKDIR /mangatra

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /mangatra/recipe.json recipe.json

RUN apt-get -yq update && apt-get -yq upgrade && apt-get -yq install \
    clang \
    make \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
	libssl-dev \
    libclang-dev \
    libopencv-dev \
    libleptonica-dev \
    libtesseract-dev \ 
    tesseract-ocr-jpn

RUN cargo chef cook --release --recipe-path recipe.json

COPY . .
RUN cargo build --release


FROM debian:bullseye-slim AS runtime
WORKDIR mangatra
COPY --from=builder /mangatra/target/release/mangatra /mangatra

RUN apt-get -yq update && apt-get -yq upgrade && apt-get -yq install \
	sudo \
	openssl \
	libopencv-dev \
	tesseract-ocr-jpn-vert

RUN useradd -ms /bin/bash mangatra
USER mangatra


RUN cd
