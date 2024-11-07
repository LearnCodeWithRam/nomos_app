CREATE DATABASE `nomos_legal`;
CREATE TABLE nomos_legal.user_account(
  user_id int NOT NULL AUTO_INCREMENT,
  user_name varchar(45) NOT NULL,
  user_email varchar(45) NOT NULL,
  user_password varchar(45) NOT NULL,
  time_stamp varchar(45) NOT NULL,
  PRIMARY KEY (user_id),
  UNIQUE KEY user_id_UNIQUE (user_id)
);

CREATE TABLE nomos_legal.chat_history (
  chat_id int NOT NULL AUTO_INCREMENT,
  user_id int NOT NULL,
  user_query mediumtext NOT NULL,
  answer1 longtext NOT NULL,
  answer2 longtext NOT NULL,
  time_stamp varchar(45) NOT NULL,
  PRIMARY KEY (chat_id),
  KEY fk_user_id_idx (user_id),
  CONSTRAINT fk_user_id FOREIGN KEY (user_id) REFERENCES user_account (user_id) ON DELETE CASCADE ON UPDATE CASCADE
);


INSERT INTO `nomos_legal`.`user_account`
(`user_id`,
 `user_name`,
 `user_email`,
 `user_password`,
 `time_stamp`)
VALUES
(1,
 'admin',
 'admin@gmail.com',
 'Admin@123',
 '2024-11-05 12:00:00');
