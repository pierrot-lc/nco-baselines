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
      pkgs.python312
      pkgs.python312Packages.venvShellHook
      pkgs.uv
    ];

    libs = [
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      cudaPackages.libcusparse
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
        uv pip install --force-reinstall torch-sparse torch-scatter "numpy==1.26" -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
