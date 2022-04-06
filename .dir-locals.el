;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((nil . ((
          projectile-project-run-cmd . "docker build -t rlcard-train -f Dockerfile.train . && docker run -v $(pwd):/app -it rlcard-train dvc repro"
        ))))
