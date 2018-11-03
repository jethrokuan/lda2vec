with import <nixpkgs> {};

(python36.buildEnv.override {
  extraLibs = with pkgs.python36Packages;
  [
    pip
    scikitlearn
    tensorflow
    numpy
    scipy
    nltk
    gensim

    # Utilities
    yapf
    python-language-server
  ];

  ignoreCollisions = true;
}).env
