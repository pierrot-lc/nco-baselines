{
  # Use `uv add jax[cuda12_local]` to install JAX in the venv.

  # Because of how NixOS works you can't install JAX with CUDA built by JAX
  # itself. Hence this flakes provides all the tiny little details to point JAX
  # to Nix's CUDA packages.

  description = "JAX devShell";

  nixConfig = {
    extra-substituters = [
      "https://cache.flox.dev"
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "flox-cache-public-1:7F4OyH7ZCnFhcze3fJdfyXYLQw/aV7GEed86nQ7IsOs="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-26.05";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };

    qsopt = rec {
      a = pkgs.stdenv.mkDerivation {
        pname = "qsopt.a";
        version = "1.0";
        src = builtins.fetchurl {
          url = "https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.a";
          sha256 = "sha256:1vilfjh5vqy4f29siqkp9ci2wrvvq6mrmrvwvswyi1ffgwy35ksx";
        };
        phases = ["installPhase"];
        installPhase = ''
          mkdir -p $out
          cp $src $out/qsopt.a
        '';
      };
      h = pkgs.stdenv.mkDerivation {
        pname = "qsopt.h";
        version = "1.0";
        src = builtins.fetchurl {
          url = "https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.h";
          sha256 = "sha256:092nwrddzxgxp2fbspfnw92wpwayf1y8kq9mrwz2dqbpppqjjxv4";
        };
        phases = ["installPhase"];
        installPhase = ''
          mkdir -p $out
          cp $src $out/qsopt.h
        '';
      };
      dir = pkgs.stdenv.mkDerivation {
        name = "QSOpt";
        unpackPhase = "true";
        buildInputs = [a h];

        installPhase = ''
          mkdir $out
          cp ${a}/qsopt.a ${h}/qsopt.h $out/
        '';
      };
    };

    concorde = pkgs.stdenv.mkDerivation {
      name = "Concorde";
      src = builtins.fetchurl {
        url = "https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz";
        sha256 = "sha256:09670naqybachcaa75jx7sh5m759jjwqh4hwx000lznmr1chlrf3";
      };
      buildInputs = [qsopt.dir pkgs.gcc pkgs.gnugrep];

      patchPhase = ''
        sed -i 's/gethostname (char \*, int);/gethostname (char \*, size_t);/' ./INCLUDE/machdefs.h
      '';
      buildPhase = ''
        mkdir build
        cd build
        CPPFLAGS="-Wno-implicit-int" ../configure  --with-qsopt=${qsopt.dir} --host=i686-pc-linux-gnu
        sed -i '/define u_char unsigned char/d' ../INCLUDE/config.h
        make
      '';
      installPhase = ''
        mkdir -p $out/bin
        cp TSP/concorde $out/bin
      '';
    };

    lkh = pkgs.stdenv.mkDerivation {
      name = "LKH-3";
      src = builtins.fetchurl {
        url = "http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz";
        sha256 = "011zmny2hmd9dmfyrfxsbzbgc82h69baqcjgb08qzra8n3kbafv9";
      };
      buildPhase = ''
        make
      '';
      installPhase = ''
        mkdir -p $out/bin
        cp LKH $out/bin
      '';
    };

    packages = [
      concorde
      lkh
      pkgs.clang
      pkgs.just
      pkgs.python312
      pkgs.python312Packages.venvShellHook
      pkgs.uv
      pkgs.openssl
      pkgs.gnumake
    ];

    cudaPackages = pkgs.cudaPackages_12;
    libs = [
      cudaPackages.cudatoolkit
      cudaPackages.libcusparse_lt
      cudaPackages.cudnn
      cudaPackages.nccl
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib

      # Where your local "lib/libcuda.so" lives. If you're not on NixOS,
      # you should provide the right path (likely another one).
      "/run/opengl-driver"
    ];

    shell = pkgs.mkShell {
      name = "jax-devshell";
      inherit packages;

      env = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
        XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit} --xla_gpu_enable_triton_gemm=false";

        # For PyTorch
        CC = "${pkgs.gcc}/bin/gcc";
        TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
        TRITON_PTXAS_PATH = "${pkgs.cudaPackages.cudatoolkit}/bin/ptxas";
      };

      venvDir = "./.venv";
      postShellHook = ''
        export PATH="$PATH:${cudaPackages.cudatoolkit}/bin"  # Add ptxas to PATH.

        uv sync
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
