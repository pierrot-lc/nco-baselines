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
      # pkgs.python313Packages.wandb
      pkgs.uv
    ];

    libs = [
      cudaPackages.cudatoolkit
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.python313

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
        CPATH = "${pkgs.python313}/include/python3.13:${pkgs.zlib.dev}/include:/home/pierrot-lc/GitHub/nco-baselines/SIL/.venv/lib/python3.13/site-packages/numpy/_core/include";

        # Specifics for PyTorch's compilation.
        CC = "${pkgs.gcc}/bin/gcc";
        TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
        TRITON_PTXAS_PATH = "${pkgs.cudaPackages.cudatoolkit}/bin/ptxas";
      };

      venvDir = "./.venv";
      postShellHook = ''
        uv sync
        cd utils/insertion; make; cd -
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
