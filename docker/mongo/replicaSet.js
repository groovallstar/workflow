config = {
  _id: "replication",
  members: [
    {_id: 0, host: "mongo1:27017", priority: 2},
    {_id: 1, host: "mongo2:27017", priority: 0},
    {_id: 2, host: "mongo3:27017", priority: 0}]
};
rs.initiate(config);
rs.conf();

use admin;
admin = db.getSiblingDB("admin");
admin.createUser({
  user: "root",
  pwd: "root",
  roles: [
    {role: "root", db: "admin"}]
});
db.getSiblingDB("admin").auth("root", "root");
