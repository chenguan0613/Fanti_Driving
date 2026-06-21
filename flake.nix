{
  description = "DevShell for Venv.";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
  in {
    devShells.${system}.default = pkgs.mkShell {
      # Add packages here.
      buildInputs = with pkgs; [
        glib
        libGL
        libx11
        libxcb
        libxext
        libxkbcommon
        libxrender
        prettier
        python313
        stdenv.cc.cc.lib
        zlib
      ];
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.glib
        pkgs.libGL
        pkgs.libxkbcommon
        pkgs.stdenv.cc.cc
        pkgs.libx11
        pkgs.libxext
        pkgs.libxrender
        pkgs.libxcb
        pkgs.zlib
      ];
      PYTHONPATH = ".";

      # Shell hooks.
      shellHook = ''
        echo "Entering the development environment!"
      '';
    };
  };
}
