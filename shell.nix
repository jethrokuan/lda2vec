with import <nixpkgs> {};

(python36.buildEnv.override {
  extraLibs = with pkgs.python36Packages;
  [ scikitlearn
    tensorflow
    numpy
    scipy
    spacy
  ];

  ignoreCollisions = true;
}).env
