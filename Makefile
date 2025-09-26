DEPLOY_DIR := deploy


all: prod

prod: setup-env pull run
dev: setup-env run-dev

restart: stop prod
redev: stop dev


run:
	cd $(DEPLOY_DIR) && docker compose up -d

run-dev:
	cd $(DEPLOY_DIR) && docker compose -f compose.yml -f compose.dev.yml up -d

stop:
	cd $(DEPLOY_DIR) && docker compose -f compose.yml -f compose.dev.yml down --rmi local

clean:
	cd $(DEPLOY_DIR) && docker compose -f compose.yml -f compose.dev.yml down --rmi local -v

pull:
	cd $(DEPLOY_DIR) && docker compose pull backend


$(DEPLOY_DIR)/.env:
	cp $(DEPLOY_DIR)/.env.example $(DEPLOY_DIR)/.env
	nano $(DEPLOY_DIR)/.env

setup-env: $(DEPLOY_DIR)/.env
