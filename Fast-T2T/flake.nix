{
  description = "PyTorch devshell";

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
    nixpkgs.url = "nixpkgs/nixos-unstable";
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

    cudaPackages = pkgs.cudaPackages_13;

    packages = [
      pkgs.gnumake
      pkgs.openssl
      pkgs.python313
      pkgs.python313Packages.venvShellHook
      pkgs.python313Packages.wandb
      pkgs.uv
    ];

    libs = [
      cudaPackages.cudatoolkit
      # cudaPackages.cudnn
      # cudaPackages.nccl
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib

      # Where your local "lib/libcuda.so" lives. If you're not on NixOS,
      # you should provide the right path (likely another one).
      "/run/opengl-driver"
    ];

    shell = pkgs.mkShell {
      name = "torch";
      inherit packages;

      env = {
        # General libs for PyTorch and Numpy.
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;

        # Specifics for PyTorch's compilation.
        CC = "${pkgs.gcc}/bin/gcc";
        TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
        TRITON_PTXAS_PATH = "${pkgs.cudaPackages.cudatoolkit}/bin/ptxas";
      };

      venvDir = "./.venv";
      postShellHook = ''
        uv sync
        uv pip install --force-reinstall torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
        cd diffusion/utils/cython_merge; python setup.py build_ext --inplace; cd -
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
