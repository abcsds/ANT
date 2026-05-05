{
  description = "Pre-randomization and reporting for the ANT experiment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in {
      apps = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          reportEnv = pkgs.python3.withPackages (ps: with ps; [
            bokeh scipy numpy pandas
          ]);
          prerandomize = pkgs.writeShellApplication {
            name = "prerandomize";
            runtimeInputs = [ pkgs.python3 ];
            text = ''
              exec python3 ${./prerandomize.py} "$@"
            '';
          };
          report = pkgs.writeShellApplication {
            name = "report";
            runtimeInputs = [ reportEnv ];
            text = ''
              exec python3 ${./report.py} "$@"
            '';
          };
        in {
          prerandomize = {
            type = "app";
            program = "${prerandomize}/bin/prerandomize";
            meta.description = "Generate ANT block lists and participant schedules";
          };
          report = {
            type = "app";
            program = "${report}/bin/report";
            meta.description = "Build per-participant ANT HTML reports under docs/";
          };
          default = self.apps.${system}.prerandomize;
        });

      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          reportEnv = pkgs.python3.withPackages (ps: with ps; [
            bokeh scipy numpy pandas
          ]);
        in {
          default = pkgs.mkShell {
            packages = [ reportEnv ];
          };
        });
    };
}
