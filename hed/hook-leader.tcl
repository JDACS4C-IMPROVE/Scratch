
# HOOK LEADER TCL
# This code runs on each leader rank,
#      i.e., once per node.

# Set a root data directory
set root $env(HOME)/data
puts "LEADER HOOK HOST: [exec hostname]"
flush stdout

# Get the leader communicator from ADLB
set comm [ adlb::comm_get leaders ]
# Get my rank among the leaders
set rank [ adlb::comm_rank $comm ]

puts "LEADER HOOK: rank: $rank"
flush stdout

if { $rank == 0 } {
  set start [ clock milliseconds ]
}

# If I am leader rank=0, discover all files to bcast:
if { $rank == 0 } {
  set pattern $env(HOME)/proj/I-UNO.jw/exp_result/*.parquet
  puts "pattern: $pattern"
  flush stdout
  set files [ glob $pattern ]
  set count [ llength $files ]
  puts "files: $count"
  flush stdout
}

# Broadcast the file list to all leaders
turbine::c::bcast $comm 0 files

puts "bcast ok"
flush stdout

# Make a node-local data directory
set LOCAL /tmp/$env(USER)/original
file mkdir $LOCAL

# Copy each file to the node-local directory
foreach f $files {
  if { $rank == 0 } {
    puts "copying: to: $LOCAL $f"
  }
  turbine::c::copy_to $comm $f $LOCAL
}

if { $rank == 0 } {
  set stop [ clock milliseconds ]
  puts "hook time: [ expr ($stop - $start) / 1000.0 ]"
  # puts [ exec ls $LOCAL ]
}
